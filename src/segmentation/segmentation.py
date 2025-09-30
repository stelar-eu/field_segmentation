import time
import geopandas as gpd
import os
import json
import numpy as np
import cv2
import glob
from pyproj.crs.crs import CRS
from shapely import unary_union
from shapely.geometry.base import BaseGeometry
from shapely.geometry import box
import shutil
import datetime as dt
import skimage.filters.rank as rank
from skimage.morphology import disk
from PIL import Image
import math

from .bands_data_package import BandsDataPackage
from .models.segmentation_unets import ResUnetA
from stelar_spatiotemporal.preprocessing.preprocessing import split_patch_into_patchlets, combine_npys_into_eopatches
from stelar_spatiotemporal.eolearn.core import EOPatch, OverwritePermission, FeatureType
from stelar_spatiotemporal.lib import *
from stelar_spatiotemporal.io import S3FileSystem

import logging
logger = logging.getLogger()


def unpack_contours(df_filename: str, threshold: float = 0.8) -> gpd.GeoDataFrame:
    """ Convert multipolygon contour row above given threshold into multiple Polygon rows. """
    df = gpd.read_file(df_filename)
    if len(df) <= 2:
        query = df[df.amax > threshold] 
        if len(query):
            return gpd.GeoDataFrame(geometry=list(query.iloc[0].geometry.geoms), crs=df.crs)
        else:
            return gpd.GeoDataFrame(geometry=[], crs=df.crs)
    raise ValueError(
        f"gdal_contour dataframe {df_filename} has {len(df)} contours, "
        f"but should have maximal 2 entries (one below and/or one above threshold)!")


# Split it into smaller patchlets (in parallel)
def patchify_segmentation_data(eop_path: str, outdir: str, patchlet_size:tuple=(1128,1128), buffer:int=100, n_jobs:int=16):
    eop_paths = [eop_path]
    multiprocess_map(func=split_patch_into_patchlets, object_list=eop_paths, n_jobs=n_jobs, 
                 patchlet_size=patchlet_size, buffer=buffer, output_dir=outdir)


def load_model(model_folder:str, tile_shape:tuple = (None,1128,1128,4)):
    """
    Load a segmentation model from a folder.
    """
    filesystem = get_filesystem(model_folder)

    # Load model config
    with filesystem.open(os.path.join(model_folder, "model_cfg.json"), "r") as f:
        model_cfg = json.load(f)

    input_shape = dict(features=tile_shape)

    # Use GPU if available, otherwise CPU
    if model_cfg.get("use_gpu", True):
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"Using GPU: {gpus[0].name}")
        else:
            logger.info("No GPU found, using CPU")

    # Load, build and compile model
    model = ResUnetA(model_cfg)
    model.build(input_shape)
    model.net.compile()

    checkpoint_dir = os.path.join(model_folder, "checkpoints")

    # If filesystem of model folder is s3, download weights into temporary folder
    if type(filesystem) == S3FileSystem:
        logger.info("Downloading model weights from S3")
        tmpdir = os.environ.get("TMPDIR", "/tmp")
        new_checkpoint_dir = os.path.join(tmpdir, 'checkpoints')
        if not os.path.exists(new_checkpoint_dir):
            filesystem.download(checkpoint_dir, tmpdir, recursive=True)
        checkpoint_dir = new_checkpoint_dir

    model.net.load_weights(os.path.join(checkpoint_dir, "model.ckpt"))

    return model


def get_patchlet_shape(patchlet_path: str):
    eop = EOPatch.load(patchlet_path, lazy_loading=True)
    return eop.data['BANDS'].shape

def smooth(array: np.ndarray, disk_size: int = 2) -> np.ndarray:
    """ Blur input array using a disk element of a given disk size """
    assert array.ndim == 2

    smoothed = rank.mean(array, disk(disk_size)).astype(np.float32)
    smoothed = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))

    # assert np.sum(~np.isfinite(smoothed)) == 0

    return smoothed

def upscale_and_rescale(array: np.ndarray, scale_factor: int = 2) -> np.ndarray:
    """ Upscale a given array by a given scale factor using bicubic interpolation """
    assert array.ndim == 2

    height, width = array.shape

    rescaled = np.array(Image.fromarray(array).resize((width * scale_factor, height * scale_factor), Image.BICUBIC))
    rescaled = (rescaled - np.min(rescaled)) / (np.max(rescaled) - np.min(rescaled))

    # assert np.sum(~np.isfinite(rescaled)) == 0

    return rescaled

def upscale_prediction(boundary: np.ndarray, scale_factor:int=2, disk_size:int=2):
    boundary = upscale_and_rescale(boundary, scale_factor=scale_factor)
    boundary = smooth(boundary, disk_size=disk_size)
    boundary = upscale_and_rescale(boundary, scale_factor=scale_factor)
    array = smooth(boundary, disk_size=disk_size * scale_factor)
    return np.expand_dims(array, axis=-1)

def smooth_optimized(array: np.ndarray, disk_size: int = 2) -> np.ndarray:
    """Optimized smoothing using OpenCV"""
    assert array.ndim == 2
    
    # Use OpenCV's blur which is typically faster than skimage
    kernel_size = disk_size * 2 + 1
    smoothed = cv2.blur(array.astype(np.float32), (kernel_size, kernel_size))
    
    # Vectorized normalization
    min_val = smoothed.min()
    max_val = smoothed.max()
    if max_val > min_val:
        smoothed = (smoothed - min_val) / (max_val - min_val)
    
    return smoothed

def upscale_and_rescale_optimized(array: np.ndarray, scale_factor: int = 2) -> np.ndarray:
    """Optimized upscaling using OpenCV"""
    assert array.ndim == 2
    
    height, width = array.shape
    new_size = (width * scale_factor, height * scale_factor)
    
    # Use OpenCV for faster interpolation
    rescaled = cv2.resize(array.astype(np.float32), new_size, interpolation=cv2.INTER_CUBIC)
    
    # Vectorized normalization
    min_val = rescaled.min()
    max_val = rescaled.max()
    if max_val > min_val:
        rescaled = (rescaled - min_val) / (max_val - min_val)
    
    return rescaled

def upscale_prediction_optimized(boundary: np.ndarray, scale_factor: int = 2, disk_size: int = 2):
    """Optimized upscaling pipeline"""
    # Pre-allocate arrays to avoid repeated memory allocation
    boundary = boundary.astype(np.float32)
    
    # First upscale
    boundary = upscale_and_rescale_optimized(boundary, scale_factor=scale_factor)
    
    # First smooth
    boundary = smooth_optimized(boundary, disk_size=disk_size)
    
    # Second upscale
    boundary = upscale_and_rescale_optimized(boundary, scale_factor=scale_factor)
    
    # Final smooth with adjusted disk size
    boundary = smooth_optimized(boundary, disk_size=disk_size * scale_factor)
    
    return np.expand_dims(boundary, axis=-1)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def to_img(rgb):
    rgb = rgb / 10000 * 5

    # Clip to [0, 1] range
    rgb = np.clip(rgb, 0, 1)

    return (rgb * 255).astype(np.uint8)

def clahe(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_img = cv2.merge((cl,a,b))
    return cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2RGB)

def unsharp_masking(rgb):
    blurred = cv2.GaussianBlur(rgb, (9,9), 10.0)
    sharpened = cv2.addWeighted(rgb, 1.5, blurred, -0.5, 0)
    return sharpened

# Sobel filter
def sobel_enhanced_rgb(rgb, alpha=0.7):
    # Convert to grayscale and compute Sobel magnitude
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = np.clip(sobel_magnitude, 0, 255).astype(np.uint8)
    
    # Normalize and stack to RGB
    sobel_rgb = cv2.cvtColor(sobel_magnitude, cv2.COLOR_GRAY2RGB)

    # Blend with original image
    enhanced = cv2.addWeighted(rgb, alpha, sobel_rgb, 1 - alpha, 0)
    return enhanced

def normalize(bands: np.ndarray) -> np.ndarray:
    """
    Normalize bands to have zero mean and unit variance.
    Bands should be in the shape (height, width, n_bands).
    """
    if bands.ndim != 3:
        raise ValueError("Input bands must be a 3D array with shape (height, width, n_bands)")

    # Calculate mean and std across spatial dimensions
    mean_stats = np.mean(bands, axis=(0, 1), keepdims=True)
    std_stats = np.std(bands, axis=(0, 1), keepdims=True)

    # Normalize bands
    norm_bands = (bands - mean_stats) / std_stats

    # Clip to [-2, 2]
    norm_bands = np.clip(norm_bands, -2, 2)

    return norm_bands

def segment_eopatch(eop_path: str, model: ResUnetA):
    # Load tile data
    eop = EOPatch.load(eop_path, lazy_loading=True)

    data = eop.data["BANDS"]
    timestamps = np.array(eop.timestamp)

    # Skip already segmented patches
    if "SEGMENTATION" in eop.data_timeless.keys():
        logger.info(f"Skipping {eop_path} as it is already segmented")
        return

    # Iterate over inference timestamps
    for timestamp, bands in zip(timestamps, data):
        # nir = bands[..., 3]  # Assuming NIR is the 4th band
        # rgb = bands[..., :3]  # Assuming RGB is the first three bands

        # # To image [0, 255] range
        # img = to_img(rgb)

        # # Enhance with clahe -> unsharp masking -> sobel filter
        # img = clahe(img)
        # img = unsharp_masking(img)
        # img = sobel_enhanced_rgb(img)

        # # Merge NIR band into RGB
        # img = np.concatenate([img, nir[..., np.newaxis]], axis=-1)

        # Normalize image
        # img = normalize(img)

        # Only normalize bands (old way)
        img = normalize(bands)

        # Segment image
        extent, boundary, distance = model.net.predict(img[np.newaxis, ...], batch_size=1)

        # Crop to original size
        extent = crop_array(extent, 12)[..., :1]
        boundary = crop_array(boundary, 12)[..., :1]
        distance = crop_array(distance, 12)[..., :1]

        # Add to eopatch (initialize features first if necessary)
        if "BOUNDARY" not in eop.data.keys():
            eop.data['BOUNDARY'] = boundary
        else:
            eop.data['BOUNDARY'] = np.concatenate([eop.data['BOUNDARY'], boundary], axis=0)

    # Combine all boundary predictions into one segmentation map
    boundary = eop.data['BOUNDARY']
    boundary_combined = combine_temporal(boundary).squeeze()

    # Upscale and smooth segmentation map
    logger.info("Upscaling and smoothing segmentation map")
    boundary_combined = upscale_prediction_optimized(boundary_combined, scale_factor=2, disk_size=2)

    # Set as segmentation map
    eop.data_timeless['SEGMENTATION'] = boundary_combined

    # Remove boundary predictions
    eop.remove_feature(FeatureType.DATA, 'BOUNDARY')

    # Save eopatch
    eop.save(eop_path, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)


def segment_patchlets(modeldir:str, patchlet_dir:str):
    # Load model
    logger.info("Loading model")
    model = load_model(modeldir)

    # Segment each patchlet
    patchlet_paths = glob.glob(os.path.join(patchlet_dir, "*"))
    for i,patchlet_path in enumerate(patchlet_paths):
        logger.info(f"{(i+1)}/{len(patchlet_paths)}: Segmenting {patchlet_path}")
        segment_eopatch(patchlet_path, model)

    # Remove model from memory
    del model

    # Stop TensorFlow session
    import tensorflow as tf
    tf.keras.backend.clear_session()


def prediction_to_tiff(eop_path:str, outdir:str):
    path_elements = eop_path.split("/")
    patchlet_name = path_elements[-1]

    tiff_dir = os.path.join(outdir, "tiffs")
    os.makedirs(tiff_dir, exist_ok=True)

    tiff_path = os.path.join(tiff_dir, patchlet_name + ".tiff")

    # Save to tiff
    export_eopatch_to_tiff(eop_path, out_path=tiff_path,feature=(FeatureType.DATA_TIMELESS, 'SEGMENTATION'))

    return tiff_path


def vectorize(eop_path:str, outdir:str, threshold:float=0.8):
    """Contour single eopatch"""

    eop_name = os.path.basename(eop_path)
    vec_path = os.path.join(outdir, eop_name + '.gpkg')

    # Skip if already vectorized
    if os.path.exists(vec_path):
        logger.info(f"Skipping {eop_name} as it is already vectorized")
        return

    # Temporarily save as tiff
    start = time.time()
    tiff_path = prediction_to_tiff(eop_path, outdir)
    logger.info(f"Done converting {eop_name} to tiff, took {time.time() - start:.2f} seconds")

    start = time.time()
    # Vectorize tiff file using gdal
    gdal_str = f"gdal_contour -of gpkg {tiff_path} {vec_path} -fl {threshold} -amin amin -amax amax -p > /dev/null"
    os.system(gdal_str)

    logger.info(f"Done vectorizing {eop_name} with threshold {threshold}, took {time.time() - start:.2f} seconds")

    start = time.time()
    # Unpack contours from tiff file
    df = unpack_contours(vec_path, threshold)
    logger.info(f"Done unpacking contours for {eop_name}, took {time.time() - start:.2f} seconds")

    # Remove too large shapes
    size = len(df)
    df = df[df.geometry.area < 100_000_000]  # Filter for smaller areas
    if len(df) < size:
        logger.info(f"Removed {size - len(df)} shapes larger than 100 million m^2 from {eop_name}")

    # Remove existing file
    try:
        os.remove(vec_path)
    except (FileNotFoundError, OSError):
        pass

    # Save as geopackage
    df.to_file(vec_path, driver='GPKG', mode='w')

    return df


def vectorize_patchlets(patchlet_dir:str, outdir:str, n_jobs:int=16, threshold:float=0.8):
    patchlet_paths = glob.glob(os.path.join(patchlet_dir, "*"))

    multiprocess_map(func=vectorize, object_list=patchlet_paths, n_jobs=n_jobs, outdir=outdir, threshold=threshold)
    
def combine_shapes(s1:BaseGeometry,s2:BaseGeometry):
    """ Combine two lists of shapes"""
    combined_list = s1 + s2

    if len(combined_list) <= 1:
        return combined_list
    else:
        return list(unary_union(combined_list).geoms)


def combine_shapes_recursive(shapes:list, left:list, right:list, crs:CRS=CRS('32630')):
    """Recursively combine shapes (divide and conquer) and save intermediate results"""

    logger.info("Combining shapes progress: {:.2f}%".format(left / len(shapes) * 100), end="\r")

    out = []
    if left == right:
        out = shapes[left]
    elif left + 1 == right:
        out = combine_shapes(shapes[left], shapes[right])
    else:
        mid = (left + right) // 2
        shapes1 = combine_shapes_recursive(shapes, left, mid, crs)
        shapes2 = combine_shapes_recursive(shapes, mid + 1, right, crs)
        out = combine_shapes(shapes1, shapes2)

    return out


def combine_shapes_overlapping(dfs:list, crs:CRS=None, min_area:int = 0, max_area:int = 500000):
    """
    Latest approach to merging on shapes based on overlapping bounding boxes of the pandas dataframes.
    """

    # Map all dfs to the same CRS, get the shapes, and the bounding boxes
    shapes = []
    boxes = []
    for df in dfs:
        if crs is None: # if no crs is given, use the first one
            crs = df.crs
        if df.crs != crs: # if crs is different, reproject to it
            df.to_crs(crs, inplace=True)
        if len(df) == 0:
            continue

        shapes += df.geometry.tolist()
        boxes.append(box(*df.geometry.total_bounds))

    logger.info("Total number of shapes before combining and filtering: ", len(shapes))

    # Get the union of intersections of all bboxes (ie the area of potential overlap)
    total_intersection = unary_union([box.intersection(other_box) for box in boxes for other_box in boxes if box != other_box])

    # Create a global dataframe with all fields
    global_df = pd.concat(dfs)

    # Keep only the 'real' fields
    global_df = global_df[global_df.geometry.area < max_area]

    # Get the intersection of all fields (i.e., keep only the fields in global_df that have some area in common with shape_df)
    isct_mask = global_df.intersects(total_intersection)

    isct_df = global_df[isct_mask]

    # Take the union of the fields in the sntersection to make sure we do not have duplicates
    isct_df = isct_df.dissolve().explode()

    # Get the difference of all fields (i.e., keep only the fields in global_df that do not have any area in common with shape_df)
    diff_df = global_df[~isct_mask]

    # Now combine the intersection and difference dataframes to get the final dataframe
    final_df = pd.concat([isct_df, diff_df], ignore_index=True)

    # Post-filter the final dataframe on area
    final_df = final_df[final_df.geometry.area > min_area]
    final_df = final_df[final_df.geometry.area < max_area]

    logger.info("Total number of shapes after combining and filtering: ", len(final_df))

    return final_df


def combine_patchlet_shapes(contours_dir:str, outpath:str, crs:CRS=None, min_area:int = 0, max_area:int = 500000):
    # Get all patchlet vector files
    vec_paths = glob.glob(os.path.join(contours_dir, "*.gpkg"))

    # Read all shapes
    dfs = [gpd.read_file(vec_path) for vec_path in vec_paths]

    # Combine shapes
    logger.info("Combining shapes", end="\r")
    df = combine_shapes_overlapping(dfs, crs, min_area, max_area)

    # Filter by area
    df = df[df.area > min_area]
    df = df[df.area < max_area]

    # Save df in temporary location
    logger.info("Saving final result", end="\r")
    tmpdir = os.environ.get("TMPDIR", "/tmp")
    temp_path = os.path.join(tmpdir, os.path.basename(outpath))
    df.to_file(temp_path, mode='w')
    
    # Ensure file is fully written by forcing filesystem sync
    with open(temp_path, 'r') as f:
        os.fsync(f.fileno())

    # Move to final location
    logger.info("Moving to final location", end="\r")
    filesystem = get_filesystem(outpath)
    filesystem.move(temp_path, outpath, overwrite=True)

    return len(df)


# def combine_npys(datadir: str, 
#                      eopatch_name:str, 
#                      dates: list = None, 
#                      feature_name:str = 'RGB', 
#                      bands:list = ['B2', 'B3', 'B4', 'B8A'], 
#                      derive_mask:bool = False, 
#                      delete_after:bool = False,
#                      partition_size:int = 10):
#     dateformat = "%Y_%m_%d"

#     # Get all the dates that have info for all bands
#     if dates is None:
#         dates = []
#         for band in bands:
#             band_dates = set()
#             banddir = os.path.join(datadir, band)
#             files = glob.glob(os.path.join(banddir, "*.npy"))
#             # Add dates to list for files with correct format
#             for file in files:
#                 try:
#                     band_dates.add(dt.datetime.strptime(os.path.basename(file).replace(".npy",""), dateformat))
#                 except ValueError:
#                     pass
#             dates = set.intersection(dates, band_dates) if len(dates) > 0 else band_dates
    
#     dates = sorted(list(dates))
    
#     if len(dates) == 0:
#         raise ValueError("No dates found with data for all bands")

#     # Check if we have all the band values for these dates
#     for date in dates:
#         for band in bands:
#             if not os.path.exists(os.path.join(datadir, band, date.strftime(dateformat) + ".npy")):
#                 raise FileNotFoundError(f"Band {band} for date {date} missing in {datadir}")
            
#     # Partition dates into groups of partition_size
#     date_partitions = [dates[i:i + partition_size] for i in range(0, len(dates), partition_size)]
#     logger.info(f"Processing {len(date_partitions)} partitions of {partition_size} dates each")

#     # Create parent directory
#     parent_dir = os.path.join(datadir, eopatch_name)
#     os.makedirs(parent_dir, exist_ok=True)

#     # Process each partition
#     for i, partition in enumerate(date_partitions):
#         logger.info(f"Processing partition {i+1}/{len(date_partitions)}", end="\r")

#         # Combine the bands into one array
#         DATA = None # SHAPE: (n_dates, img_h, img_w, n_bands)
#         MASK = None # SHAPE: (n_dates, img_h, img_w)

#         for date in partition:
#             # Create a local array
#             band_array = [np.load(os.path.join(datadir, band, date.strftime(dateformat) + ".npy")) for band in bands]

#             # Stack the bands
#             data = np.stack(band_array, axis=-1)[np.newaxis, ...]

#             # Add to global array
#             if DATA is None:
#                 DATA = data
#             else:
#                 DATA = np.concatenate([DATA, data], axis=0)

#             # Derive mask
#             if derive_mask:
#                 mask = data[..., -1] > 0
#                 mask = mask.astype(np.uint8)[..., np.newaxis]
#                 if MASK is None:
#                     MASK = mask
#                 else:
#                     MASK = np.concatenate([MASK, mask], axis=0)

#         # Get bboxes from band folders
#         bboxes = []
#         for band in bands:
#             bbox_path = os.path.join(datadir, band, 'bbox.txt')
#             bboxes.append(bbox_from_file(bbox_path))

#         # Check if all bboxes are the same
#         bbox = bboxes[0]
#         for b in bboxes:
#             if b != bbox:
#                 raise ValueError("Bounding boxes for each band are not the same")
        
#         # Create eopatch
#         eopatch = EOPatch()
#         eopatch.data[feature_name] = DATA
#         eopatch.bbox = bbox
#         eopatch.timestamp = partition

#         # Add mask if necessary
#         if derive_mask:
#             eopatch.mask['IS_DATA'] = MASK

#         # Save eopatch
#         logger.info(f"Saving eopatch {i+1}/{len(date_partitions)}", end="\r")
#         begin_date = partition[0].strftime(dateformat)
#         eopatch.save(os.path.join(parent_dir, f"{eopatch_name}_{begin_date}"), overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

#     # (Optional) Delete all the individual files
#     if delete_after:
#         for date in dates:
#             for band in bands:
#                 os.remove(os.path.join(datadir, band, date.strftime(dateformat) + ".npy"))
#             os.remove(os.path.join(datadir, "bbox_" + date.strftime(dateformat) + ".txt"))

#     return eopatch



def combine_rgb_npys_into_eopatch(bands_data_package:BandsDataPackage, outdir:str, dates:list = None) -> str:
    DATE_FORMAT = "%Y_%m_%d"

    if dates is None:
        dates = [dt.datetime.strptime(os.path.basename(pair[0]).replace(".RAS",""), DATE_FORMAT) for pair in bands_data_package.B2_package.path_pairs]

    if len(dates) > 1000:
        raise ValueError("Too many dates to segment")

    bands = bands_data_package.tolist()

    # Check if all bands have the necessary segment dates
    for band_data_package in bands:
        for segment_date in dates:
            if not os.path.exists(os.path.join(band_data_package.BAND_DIR, segment_date.strftime(f"{DATE_FORMAT}.npy"))):
                raise ValueError(f"Date {segment_date} not found in {band_data_package.BAND_DIR}")

    # Combine the segment dates into one eopatch per band
    logger.info("Combining segment dates into one eopatch per band")
    eop_paths = []
    for band_data_package in bands:
        bbox = load_bbox(os.path.join(band_data_package.BAND_DIR, "bbox.pkl"))

        outpath = os.path.join(outdir, band_data_package.BAND_NAME + "_eopatch")
        eop_paths += [outpath]

        npy_paths = [os.path.join(band_data_package.BAND_DIR, date.strftime(f"{DATE_FORMAT}.npy")) for date in dates]

        combine_npys_into_eopatches(
            npy_paths=npy_paths,
            outpath=outpath,
            feature_name='BANDS',
            bbox=bbox,
            dates=dates,
            delete_after=False,
            partition_size=1000
        )

    # Now combine the eopatches into one eopatch
    logger.info("Combining eopatches into one eopatch")
    base_eopatch = None
    for eop_path in eop_paths:
        eop = EOPatch.load(eop_path)
        if base_eopatch is None:
            base_eopatch = eop
        else:
            base_eopatch.data['BANDS'] = np.concatenate((base_eopatch.data['BANDS'], eop.data['BANDS']), axis=-1)
        
    # Save the eopatch
    logger.info("Saving eopatch")
    outpath = os.path.join(outdir, "segment_eopatch")
    base_eopatch.save(outpath, overwrite_permission=OverwritePermission.OVERWRITE_FEATURES)

    return outpath