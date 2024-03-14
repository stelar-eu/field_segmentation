import os
import glob
import sys
import datetime as dt
from sentinelhub import CRS
from typing import List, Text, Tuple
from stelar_spatiotemporal.preprocessing.preprocessing import combine_npys_into_eopatches
from stelar_spatiotemporal.preprocessing.vista_preprocessing import unpack_vista_reflectance
from stelar_spatiotemporal.segmentation.bands_data_package import BandsDataPackage
from stelar_spatiotemporal.segmentation.segmentation import *
from stelar_spatiotemporal.eolearn.core import OverwritePermission
import warnings
warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description="Segment a Sentinel-2 tile using a pre-trained ResUNet model")

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
        print("Dates should be given in the format YYYY-MM-DD")
        sys.exit(1)

    return sdates

def cleanup(*args):
    for todel_path in args:
        if os.path.exists(todel_path):
            print("Deleting {}".format(todel_path))
            os.system("rm -rf {}".format(todel_path))


def segmentation_pipeline(bands_data_package:BandsDataPackage, out_path:Text, model_path:Text, sdates:List[dt.datetime] = None, do_cleanup:bool = True):
    """
    Pipeline for segmenting a Sentinel-2 tile using a pre-trained ResUNet model.

    Parameters
    ----------
    bands_data_package : BandsDataPackage
        Data package containing the paths to the RAS files with RGBNIR bands
    out_path : Text
        Path to the output shapefile
    model_path : Text
        Path to the pre-trained ResUNet model
    sdates : List[dt.datetime], optional
        List of dates to segment, by default None (segment all overlapping dates in the RAS files)
    """
    
    TMPDIR = '/tmp'
        
    # 1. # Unpacks RAS and RHD files into numpy arrays
    print("1. Unpacking RAS files...")
    npy_dir = os.path.join(TMPDIR, "npys")
    unpack_vista_reflectance(bands_data_package, outdir=npy_dir, crs=CRS(32630)) 

    # Create band data package with local paths
    b2_path = os.path.join(npy_dir, "B2")
    b3_path = os.path.join(npy_dir, "B3")
    b4_path = os.path.join(npy_dir, "B4")
    b8_path = os.path.join(npy_dir, "B8A")
    npy_data_package = BandsDataPackage(b2_path, b3_path, b4_path, b8_path, file_extension="npy")

    # Get the dates if it's not given
    if sdates is None:
        sdates = ",".join([d.replace(".npy", "").replace("_", "-") for d in os.listdir(npy_data_package.B2_package.BAND_PATH) if d.endswith(".npy")])
        sdates = parse_sdates(sdates)

    # 2. Combining the images into one eopatch
    print("2. Combining images into one eopatch...")
    eopatches_dir = os.path.join(TMPDIR, "segment_eopatch")
    eop_path = combine_rgb_npys_into_eopatch(npy_data_package, outdir=eopatches_dir, dates=sdates)

    # 3. Splitting the eopatch into patchlets
    print("3. Splitting eopatch into patchlets...")
    plet_dir = os.path.join(TMPDIR, "patchlets")

    # Decide buffer and patchlet size; if full tile is smaller than patchlet, make buffer 0, else 100
    patchlet_shape = (1128,1128)
    buffer = 100

    eop_shape = EOPatch.load(eop_path).data['BANDS'].shape
    if eop_shape[1] <= patchlet_shape[0] and eop_shape[2] <= patchlet_shape[1]:
        buffer = 0

    patchify_segmentation_data(eop_path, outdir=plet_dir, n_jobs=1, patchlet_size=patchlet_shape, buffer=buffer)

    # 4. Run segmentation
    print("4. Running segmentation...")
    segment_patchlets(model_path, plet_dir)

    # 5. Vectorize segmentation
    print("5. Vectorizing segmentation...")
    vecs_dir = os.path.join(TMPDIR, "contours")
    vectorize_patchlets(plet_dir, outdir=vecs_dir)

    # 6. Combine patchlets shapes single shapefile
    print("6. Combining patchlet shapes into single shapefile...")
    combine_patchlet_shapes(vecs_dir, out_path)

    # 7. Cleanup
    if do_cleanup:
        print("7. Cleaning up...")
        cleanup(npy_dir, eopatches_dir, plet_dir, vecs_dir)


parser.add_argument("-b2", "--b2_path", 
                    type=str, 
                    required=True,
                    help="Path to the folder of RAS files containing the B2 band.")
parser.add_argument("-b3", "--b3_path",
                    type=str,
                    required=True,
                    help="Path to the folder of RAS files containing the B3 band.")
parser.add_argument("-b4", "--b4_path",
                    type=str,
                    required=True,
                    help="Path to the folder of RAS files containing the B4 band.")
parser.add_argument("-b8", "--b8_path",
                    type=str,
                    required=True,
                    help="Path to the folder of RAS files containing the B8A band.")
parser.add_argument("-o", "--output_path",
                    type=str,
                    required=True,
                    help="Path of the output shapefile. Should end with .shp or .gpkg.")
parser.add_argument("-m", "--model_path",
                    type=str,
                    required=True,
                    help="Path to the pre-trained ResUNet model.")
parser.add_argument("-sdates", "--segment_dates",
                    type=str,
                    required=False,
                    default=None,
                    help="Comma-separated list of dates to segment in the format YYYY-MM-DD. If not given, all overlapping dates in the RAS files will be segmented.")
parser.add_argument("--MINIO_ACCESS_KEY",
                    type=str,
                    required=False,
                    default=None,
                    help="Access key for the MinIO server. Required if the input and output paths are on MinIO (i.e., start with 's3://').")
parser.add_argument("--MINIO_SECRET_KEY",
                    type=str,
                    required=False,
                    default=None,
                    help="Secret key for the MinIO server. Required if the input and output paths are on MinIO (i.e., start with 's3://').")
parser.add_argument("--MINIO_ENDPOINT_URL",
                    type=str,
                    required=False,
                    default=None,
                    help="Endpoint URL for the MinIO server. Required if the input and output paths are on MinIO (i.e., start with 's3://').")
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        b2_path = "s3://stelar-spatiotemporal/RGB_small/B2"
        b3_path = "s3://stelar-spatiotemporal/RGB_small/B3"
        b4_path = "s3://stelar-spatiotemporal/RGB_small/B4"
        b8_path = "s3://stelar-spatiotemporal/RGB_small/B8"
        output_path = "s3://stelar-spatiotemporal/fields_test.gpkg"
        model_path = "s3://stelar-spatiotemporal/resunet-a_fold-0_2023-03-27-09-29-38"
        sdates = "2020-07-27"
        MINIO_ACCESS_KEY = "minioadmin"
        MINIO_SECRET_KEY = "minioadmin"
        MINIO_ENDPOINT_URL = "http://localhost:9000"
    else:
        args = parser.parse_args()
        b2_path = args.b2_path
        b3_path = args.b3_path
        b4_path = args.b4_path
        b8_path = args.b8_path
        output_path = args.output_path
        model_path = args.model_path
        sdates = args.segment_dates
        MINIO_ACCESS_KEY = args.MINIO_ACCESS_KEY
        MINIO_SECRET_KEY = args.MINIO_SECRET_KEY
        MINIO_ENDPOINT_URL = args.MINIO_ENDPOINT_URL

    # Check dependencies between arguments
    isminio = any([p.startswith("s3://") for p in [b2_path, b3_path, b4_path, b8_path, output_path, model_path]])
    nocred = MINIO_ACCESS_KEY is None or MINIO_SECRET_KEY is None or MINIO_ENDPOINT_URL is None
    if isminio and nocred:
        raise ValueError("Access and secret keys are required if any path is on MinIO.")
    
    # Set the environment variables
    if isminio:
        os.environ["MINIO_ACCESS_KEY"] = MINIO_ACCESS_KEY
        os.environ["MINIO_SECRET_KEY"] = MINIO_SECRET_KEY
        os.environ["MINIO_ENDPOINT_URL"] = MINIO_ENDPOINT_URL

    # Create bands data bundle
    bands_data_package = BandsDataPackage(b2_path=b2_path,
                                        b3_path=b3_path,
                                        b4_path=b4_path,
                                        b8_path=b8_path,
                                        file_extension="RAS")
    
    # Parse the segment dates
    sdates = parse_sdates(sdates)
    
    # Run the pipeline
    segmentation_pipeline(
        bands_data_package, 
        output_path, 
        model_path, 
        sdates,
        do_cleanup=False)
    