import os
import json
import time
import sys
import argparse
import datetime as dt
from sentinelhub import CRS
from typing import List, Text, Tuple
from stelar_spatiotemporal.preprocessing.preprocessing import combine_npys_into_eopatches
from stelar_spatiotemporal.preprocessing.vista_preprocessing import unpack_vista_reflectance
from stelar_spatiotemporal.segmentation.bands_data_package import BandsDataPackage, BandDataPackage
from stelar_spatiotemporal.segmentation.segmentation import *
from stelar_spatiotemporal.eolearn.core import OverwritePermission
import warnings
warnings.filterwarnings("ignore")


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


def segmentation_pipeline(bands_data_package:BandsDataPackage, out_path:Text, model_path:Text, crs:CRS, sdates:List[dt.datetime] = None, vectorization_threshold:float = 0.8, tmpdir:Text = '/tmp') -> dict:
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
    vectorization_threshold : float, optional
        Threshold for vectorization, by default 0.8
    tmpdir : Text, optional
        Directory for temporary files, by default '/tmp'
    """
    total_start = time.time()
    partial_times = []
    
    TMPDIR = tmpdir
        
    # 1. # Unpacks RAS and RHD files into numpy arrays
    start = time.time()

    print("1. Unpacking RAS files...")
    npy_dir = os.path.join(TMPDIR, "npys")
    unpack_vista_reflectance(bands_data_package, outdir=npy_dir, crs=crs) 

    # Create band data package with local paths
    b2_path = os.path.join(npy_dir, "B2")
    b3_path = os.path.join(npy_dir, "B3")
    b4_path = os.path.join(npy_dir, "B4")
    b8_path = os.path.join(npy_dir, "B8A")
    npy_data_package = BandsDataPackage(b2_path, b3_path, b4_path, b8_path, file_extension="npy")

    # Get the dates if it's not given
    if sdates is None:
        sdates = ",".join([d.replace(".npy", "").replace("_", "-") for d in os.listdir(npy_data_package.B2_package.BAND_DIR) if d.endswith(".npy")])
        sdates = parse_sdates(sdates)

    partial_times.append({
        "step": "Unpacking RAS files",
        "runtime": time.time() - start
    })

    # 2. Combining the images into one eopatch
    start = time.time()

    print("2. Combining images into one eopatch...")
    eopatches_dir = os.path.join(TMPDIR, "segment_eopatch")
    eop_path = combine_rgb_npys_into_eopatch(npy_data_package, outdir=eopatches_dir, dates=sdates)

    partial_times.append({
        "step": "Combining images into eopatch",
        "runtime": time.time() - start
    })

    # 3. Splitting the eopatch into patchlets
    start = time.time()

    print("3. Splitting eopatch into patchlets...")
    plet_dir = os.path.join(TMPDIR, "patchlets")

    # Decide buffer and patchlet size; if full tile is smaller than patchlet, make buffer 0, else 100
    patchlet_shape = (1128,1128)
    buffer = 100

    eop_shape = EOPatch.load(eop_path).data['BANDS'].shape
    if eop_shape[1] <= patchlet_shape[0] and eop_shape[2] <= patchlet_shape[1]:
        buffer = 0

    patchify_segmentation_data(eop_path, outdir=plet_dir, n_jobs=1, patchlet_size=patchlet_shape, buffer=buffer)

    partial_times.append({
        "step": "Splitting eopatch into patchlets",
        "runtime": time.time() - start
    })

    # 4. Run segmentation
    start = time.time()

    print("4. Running segmentation...")
    segment_patchlets(model_path, plet_dir)

    partial_times.append({
        "step": "Running segmentation",
        "runtime": time.time() - start
    })

    # 5. Vectorize segmentation
    start = time.time()

    print("5. Vectorizing segmentation...")
    vecs_dir = os.path.join(TMPDIR, "contours")
    vectorize_patchlets(plet_dir, outdir=vecs_dir, threshold=vectorization_threshold)

    partial_times.append({
        "step": "Vectorizing segmentation",
        "runtime": time.time() - start
    })

    # 6. Combine patchlets shapes single shapefile
    start = time.time()

    print("6. Combining patchlet shapes into single shapefile...")
    n_fields = combine_patchlet_shapes(vecs_dir, out_path, crs=crs)

    partial_times.append({
        "step": "Combining patchlet shapes into single shapefile",
        "runtime": time.time() - start
    })

    # 7. Create the output response
    response = {
        "message": "Segmentation pipeline completed successfully.",
        "output": [
            {
                "path": out_path,
                "type": "Output shapefile"
            }
        ],
        "metrics": {
            "b2_path": b2_path,
            "b3_path": b3_path,
            "b4_path": b4_path,
            "b8_path": b8_path,
            "model_path": model_path,
            "sdates": [d.strftime("%Y-%m-%d") for d in sdates],
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
        input_json_path = "/home/jens/ownCloud/Documents/3.Werk/0.TUe_Research/0.STELAR/0.VISTA/VISTA_workbench/src/modules/segmentation/resources/input.json"
    else:
        input_json_path = args.input_json
    
    if not args.output_json:
        output_json_path = "/home/jens/ownCloud/Documents/3.Werk/0.TUe_Research/0.STELAR/0.VISTA/VISTA_workbench/src/modules/segmentation/resources/output.json"
    else:
        output_json_path = args.output_json

    # Read and parse the input JSON file
    with open(input_json_path, "r") as f:
        input_json = json.load(f)

    # Check for result key in the JSON structure
    if "result" in input_json:
        input_json = input_json["result"]

    # Necessary arguments - extract paths from the RGB list
    try:
        # Get all input files from the RGB list
        input_files = input_json["input"]["RGB"]
        
        # Create bands data package using the from_file_list method
        bands_data_package = BandsDataPackage.from_file_list(input_files, file_extension="RAS")
        
    except Exception as e:
        raise ValueError(f"Passed input paths are not correct. Please see the documentation for the suggested input format. Error: {e}")
    
    # More required arguments
    try:
        output_path = input_json["output"]["segmentation_map"]
        model_path = input_json["parameters"]["model_path"]
        sdates = input_json["parameters"].get("sdates")
        
        # Parse CRS if provided, otherwise default to WGS84
        crs_value = input_json["parameters"].get("crs")
        if crs_value:
            if crs_value.lower() == "wgs84":
                crs = CRS.WGS84
            elif crs_value.lower() == "pop_web":
                crs = CRS.POP_WEB
            else:
                try:
                    # Try to parse as EPSG code
                    epsg_code = int(crs_value.replace("epsg:", "").strip())
                    crs = CRS(epsg_code)
                except:
                    print(f"Unknown CRS value: {crs_value}, defaulting to WGS84")
                    crs = CRS.WGS84
        else:
            crs = CRS.WGS84
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
        except Exception as e:
            raise ValueError(f"Access and secret keys are required if any path is on MinIO. Error: {e}")

    # Parse the segment dates
    sdates = parse_sdates(sdates)
    
    # Check if tmpdir is provided in command line or parameters
    # Command line argument has priority over JSON parameter
    tmpdir = args.tmpdir if args.tmpdir else input_json["parameters"].get("tmpdir", '/tmp')
    
    # Run the pipeline
    response = segmentation_pipeline(
        bands_data_package, 
        output_path, 
        model_path, 
        crs=crs,  # Use parsed CRS
        sdates=sdates,
        vectorization_threshold=input_json["parameters"].get("vectorization_threshold", 0.8), # Default to 0.8 if not provided
        tmpdir=tmpdir
        )
    
    # Write the output JSON file
    with open(output_json_path, "w") as f:
        json.dump(response, f)

    print(response)

