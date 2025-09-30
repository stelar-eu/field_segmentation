import os
import json
import time
import sys
import argparse
import datetime as dt
import rasterio
import numpy as np
from minio import Minio
from rasterio.crs import CRS as RioCRS
from sentinelhub import CRS, BBox
from typing import List, Text, Tuple
from stelar_spatiotemporal.lib import get_filesystem
from stelar_spatiotemporal.preprocessing.preprocessing import combine_npys_into_eopatches
from stelar_spatiotemporal.eolearn.core import EOPatch, FeatureType
from src.segmentation.segmentation import *
from stelar_spatiotemporal.eolearn.core import OverwritePermission

import logging

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

# Remove all warnings
import warnings
warnings.filterwarnings("ignore")

# Suppress TensorFlow GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def parse_sdates(s) -> List[dt.datetime]:
    """
    Parse the segment dates from the arguments
    """

    if s is None:
        return None

    # Look if specific dates are given
    sdates = s.split(",")

    # Parse the dates
    try:
        sdates = [dt.datetime.strptime(date, "%Y-%m-%d") for date in sdates]
    except ValueError:
        logger.error("Dates should be given in the format YYYY-MM-DD")
        sys.exit(1)

    return sdates

def cleanup(*args):
    for todel_path in args:
        if os.path.exists(todel_path):
            logger.info("Deleting {}".format(todel_path))
            os.system("rm -rf {}".format(todel_path))

def setup_client():
    """
    Setup the MinIO client using environment variables.
    """
    try:
        url = os.environ["MINIO_ENDPOINT_URL"]
        access_key = os.environ["MINIO_ACCESS_KEY"]
        secret_key = os.environ["MINIO_SECRET_KEY"]
        session_token = os.environ.get("MINIO_SESSION_TOKEN", None)

        minio_client = Minio(
            url.replace("https://", "").replace("http://", ""),
            access_key=access_key,
            secret_key=secret_key,
            session_token=session_token if session_token else None,
            secure= url.startswith("https://")
        )
        return minio_client
    except KeyError as e:
        raise ValueError(f"Missing environment variable: {e}")


def segmentation_pipeline(tif_path: Text, out_path: Text, model_path: Text, vectorization_threshold: float = 0.8, tmpdir: Text = '/tmp') -> dict:
    """
    Pipeline for segmenting a Sentinel-2 tile using a pre-trained ResUNet model.

    Parameters
    ----------
    tif_path : Text
        Path to the input TIF file containing RGBNIR bands
    out_path : Text
        Path to the output shapefile
    model_path : Text
        Path to the pre-trained ResUNet model
    vectorization_threshold : float, optional
        Threshold for vectorization, by default 0.8
    tmpdir : Text, optional
        Directory for temporary files, by default '/tmp'
    """
    total_start = time.time()
    partial_times = {}
    
    os.environ["TMPDIR"] = tmpdir
        
    # 1. Read TIF file and create eopatch
    start = time.time()

    logger.info("1. Reading TIF file and creating eopatch...")
    
    # Download the TIF file if it is on MinIO
    if tif_path.startswith("s3://"):
        bucket_name, object_name = tif_path.replace("s3://", "").split('/', 1)
        local_tif_path = os.path.join(tmpdir, os.path.basename(object_name))
        if not os.path.exists(local_tif_path):
            client = setup_client()
            logger.info(f"Downloading {tif_path} to {local_tif_path}...")
            client.fget_object(bucket_name, object_name, local_tif_path)
        tif_path = local_tif_path
    
    with rasterio.open(tif_path) as src:
        # Read the bands (assuming RGBNIR order: B2, B3, B4, B8A)
        bands_data = src.read()  # Shape: (bands, height, width)

        if bands_data.shape[0] == 9:
            bands_data = bands_data[[0,1,2,6], :, :]  # Select B2, B3, B4, B8A (RGBNIR)
        elif bands_data.shape[0] != 4:
            raise ValueError(f"Expected 4 bands (RGBNIR), but found {bands_data.shape[0]} bands in the TIF file.")
        
        # Check if width and height are at least 1128x1128
        if bands_data.shape[1] < 1128 or bands_data.shape[2] < 1128:
            raise ValueError(f"Input TIF image is too small. Minimum size is 1128x1128 pixels, but got {bands_data.shape[1]}x{bands_data.shape[2]} pixels.")
        
        bands_data *= 100 # To match the old RAS format scaling

        logger.info(f"Read {bands_data.shape[0]} bands from TIF file: {tif_path}")

        crs = CRS(src.crs.to_epsg()) if src.crs else CRS.WGS84        
        # Get date from filename or metadata (assuming format contains date)
        filename = os.path.basename(tif_path)
        # Extract date from filename pattern like S2B_30TYQ2BP_220614_R.TIF
        try:
            date_str = filename.split('_')[2]  # Gets '220614'
            # Convert YYMMDD to datetime
            year = 2000 + int(date_str[:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            acquisition_date = dt.datetime(year, month, day)
        except:
            # Fallback to current date if parsing fails
            acquisition_date = dt.datetime.now()
    
    # Create eopatch directly from the TIF data
    eopatches_dir = os.path.join(tmpdir, "segment_eopatch")
    os.makedirs(eopatches_dir, exist_ok=True)
    eop_path = os.path.join(eopatches_dir, "eopatch")
    
    # Transpose bands data to match eopatch format (time, height, width, bands)
    bands_data_transposed = np.transpose(bands_data, (1, 2, 0))  # (height, width, bands)
    bands_data_with_time = np.expand_dims(bands_data_transposed, axis=0)  # (1, height, width, bands)
    
    # Create eopatch
    eopatch = EOPatch()
    eopatch.data['BANDS'] = bands_data_with_time
    eopatch.timestamp = [acquisition_date]
    eopatch.bbox = BBox(bbox=src.bounds, crs=crs)
    eopatch.save(eop_path, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    # Remove eopatch from memory to free up resources
    del eopatch

    partial_times["tif_reading_eopatch_creation"] = time.time() - start

    # 2. Splitting the eopatch into patchlets
    start = time.time()

    logger.info("2. Splitting eopatch into patchlets...")
    plet_dir = os.path.join(tmpdir, "patchlets")

    # Decide buffer and patchlet size; if full tile is smaller than patchlet, make buffer 0, else 100
    patchlet_shape = (1128,1128)
    buffer = 100

    eop_shape = EOPatch.load(eop_path).data['BANDS'].shape
    if eop_shape[1] <= patchlet_shape[0] and eop_shape[2] <= patchlet_shape[1]:
        buffer = 0

    patchify_segmentation_data(eop_path, outdir=plet_dir, n_jobs=1, patchlet_size=patchlet_shape, buffer=buffer)

    partial_times["eopatch_split_into_patchlets"] = time.time() - start

    # 3. Run segmentation
    start = time.time()

    logger.info("3. Running segmentation...")
    segment_patchlets(model_path, plet_dir)

    partial_times["segmentation"] = time.time() - start

    # 4. Vectorize segmentation
    start = time.time()

    logger.info("4. Vectorizing segmentation...")
    vecs_dir = os.path.join(tmpdir, "contours")
    vectorize_patchlets(plet_dir, outdir=vecs_dir, threshold=vectorization_threshold)

    partial_times["vectorization"] = time.time() - start

    # 5. Combine patchlets shapes single shapefile
    start = time.time()

    logger.info("5. Combining patchlet shapes into single shapefile...")
    n_fields = combine_patchlet_shapes(vecs_dir, out_path)

    partial_times["combine_patchlet_shapes"] = time.time() - start

    # 6. Create the output response
    response = {
        "message": "Segmentation pipeline completed successfully.",
        "output": {
            "segmentation_map": out_path,
        },
        "status": "success",
        "metrics": {
            "tif_path": tif_path,
            "model_path": model_path,
            "acquisition_date": acquisition_date.strftime("%Y-%m-%d"),
            "crs": str(crs),
            "total_runtime": time.time() - total_start,
            "partial_runtimes": partial_times,
            "n_fields": n_fields,
        }
    }

    return response

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Field Segmentation Pipeline")
    parser.add_argument("input_json", nargs="?", help="Path to input JSON file")
    parser.add_argument("output_json", nargs="?", help="Path to output JSON file")
    parser.add_argument("--tmpdir", type=str, help="Directory for temporary files", default="/tmp")
    
    args = parser.parse_args()
    
    # Set default values for input and output JSON paths if not provided
    if not args.input_json:
        input_json_path = "resources/input_small.json"
    else:
        input_json_path = args.input_json
    
    if not args.output_json:
        output_json_path = "resources/output.json"
    else:
        output_json_path = args.output_json

    try:
        # Read and parse the input JSON file
        with open(input_json_path, "r") as f:
            input_json = json.load(f)

        # Check for result key in the JSON structure
        if "result" in input_json:
            input_json = input_json["result"]

        # Get the TIF file path
        try:
            tif_path = input_json["input"]["RGB"][0]
        except Exception as e:
            raise ValueError(f"TIF path not found in input. Error: {e}")
        
        # More required arguments
        try:
            output_path = input_json["output"]["segmentation_map"]
            model_path = input_json["parameters"]["model_path"]
        except Exception as e:
            raise ValueError(f"Missing required arguments. Please see the documentation for the suggested input format. Error: {e}")
        
        # Check if minio credentials are provided
        if "minio" in input_json:
            try:
                id = input_json["minio"]["id"]
                key = input_json["minio"]["key"]
                token = input_json["minio"].get("skey")
                url = input_json["minio"]["endpoint_url"]

                os.environ["MINIO_ACCESS_KEY"] = id
                os.environ["MINIO_SECRET_KEY"] = key
                os.environ["MINIO_ENDPOINT_URL"] = url

                # If token is provided, set it as well
                if token:
                    os.environ["MINIO_SESSION_TOKEN"] = token

                os.environ["AWS_ACCESS_KEY_ID"] = id
                os.environ["AWS_SECRET_ACCESS_KEY"] = key
                if token:
                    os.environ["AWS_SESSION_TOKEN"] = token
            except Exception as e:
                raise ValueError(f"Access and secret keys are required if any path is on MinIO. Error: {e}")
        
        # Check if tmpdir is provided in command line or parameters
        # Command line argument has priority over JSON parameter
        tmpdir = args.tmpdir if args.tmpdir else input_json["parameters"].get("tmpdir", '/tmp')
        
        # Run the pipeline
        response = segmentation_pipeline(
            tif_path, 
            output_path, 
            model_path, 
            vectorization_threshold=input_json["parameters"].get("vectorization_threshold", 0.8), # Default to 0.8 if not provided
            tmpdir=tmpdir
        )
        
        # Write the output JSON file
        with open(output_json_path, "w") as f:
            json.dump(response, f)

        logger.info(response)

    except Exception as e:
        error_response = {
            "message": "An error occurred during the segmentation pipeline.",
            "error": str(e),
            "status": "failed"
        }
        # Write the error JSON file
        with open(output_json_path, "w") as f:
            json.dump(error_response, f)

        logger.error(error_response)