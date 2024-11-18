import os
import json
import time
import sys
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


def segmentation_pipeline(bands_data_package:BandsDataPackage, out_path:Text, model_path:Text, sdates:List[dt.datetime] = None):
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
    total_start = time.time()
    partial_times = []
    
    TMPDIR = '/tmp'
        
    # 1. # Unpacks RAS and RHD files into numpy arrays
    start = time.time()

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
    vectorize_patchlets(plet_dir, outdir=vecs_dir)

    partial_times.append({
        "step": "Vectorizing segmentation",
        "runtime": time.time() - start
    })

    # 6. Combine patchlets shapes single shapefile
    start = time.time()

    print("6. Combining patchlet shapes into single shapefile...")
    combine_patchlet_shapes(vecs_dir, out_path)

    partial_times.append({
        "step": "Combining patchlet shapes into single shapefile",
        "runtime": time.time() - start
    })

    # Read final shapefile to get number of fields
    tmp_path = os.path.join("/tmp", os.path.basename(out_path))
    n_fields = gpd.read_file(tmp_path).shape[0]
    
    # 7. Create the output response
    response = {
        "message": "Segmentation pipeline completed successfully.",
        "output": [
            {
                "path": output_path,
                "type": "Output shapefile"
            }
        ],
        "metrics": {
            "b2_path": b2_path,
            "b3_path": b3_path,
            "b4_path": b4_path,
            "b8_path": b8_path,
            "model_path": model_path,
            "sdates": sdates,
            "total_runtime": time.time() - total_start,
            "partial_runtimes": partial_times,
            "n_fields": n_fields,
        }
    }

    return response

if __name__ == "__main__":
    if len(sys.argv) < 3: # If no arguments are given, use the default values
        input_json_path = "/home/jens/ownCloud/Documents/3.Werk/0.TUe_Research/0.STELAR/0.VISTA/VISTA_workbench/src/modules/segmentation/resources/input.json"
        output_json_path = "/home/jens/ownCloud/Documents/3.Werk/0.TUe_Research/0.STELAR/0.VISTA/VISTA_workbench/src/modules/segmentation/resources/output.json"
    else:
        input_json_path = sys.argv[1]
        output_json_path = sys.argv[2]


    # Read and parse the input JSON file
    with open(input_json_path, "r") as f:
        input_json = json.load(f)

    # Necessary arguments
    try:
        input_paths = input_json["input"]

        # Get the b2 path
        b2_paths = [p for p in input_paths if "B2" in p['name']]
        b2_path = b2_paths[0]['path']

        # Get the b3 path
        b3_paths = [p for p in input_paths if "B3" in p['name']]
        b3_path = b3_paths[0]['path']

        # Get the b4 path
        b4_paths = [p for p in input_paths if "B4" in p['name']]
        b4_path = b4_paths[0]['path']

        # Get the b8 path
        b8_paths = [p for p in input_paths if "B8" in p['name']]
        b8_path = b8_paths[0]['path']
    except Exception as e:
        raise ValueError("Passed input paths are not correct. Please see the documentation for the suggested input format. Error: {}".format(e))
    
    # More required arguments
    try:
        output_path = input_json["parameters"]["output_path"]
        model_path = input_json["parameters"]["model_path"]
        sdates = input_json["parameters"]["sdates"]
    except Exception as e:
        raise ValueError("Missing required arguments. Please see the documentation for the suggested input format. Error: {}".format(e))

    
    # Check if minio credentials are provided
    if "minio" in input_json:
        try:
            id = input_json["minio"]["id"]
            key = input_json["minio"]["key"]
            url = input_json["minio"]["endpoint_url"]

            os.environ["MINIO_ACCESS_KEY"] = id
            os.environ["MINIO_SECRET_KEY"] = key
            os.environ["MINIO_ENDPOINT_URL"] = url

            os.environ["AWS_ACCESS_KEY_ID"] = id
            os.environ["AWS_SECRET_ACCESS_KEY"] = key

            # Add s3:// to the input and output paths
            b2_path = "s3://" + b2_path
            b3_path = "s3://" + b3_path
            b4_path = "s3://" + b4_path
            b8_path = "s3://" + b8_path
            output_path = "s3://" + output_path
            model_path = "s3://" + model_path
        except Exception as e:
            raise ValueError("Access and secret keys are required if any path is on MinIO. Error: {}".format(e))

    # Create bands data bundle
    bands_data_package = BandsDataPackage(b2_path=b2_path, b3_path=b3_path, b4_path=b4_path, b8_path=b8_path, file_extension="RAS")
    
    # Parse the segment dates
    sdates = parse_sdates(sdates)
    
    # Run the pipeline
    response = segmentation_pipeline(
        bands_data_package, 
        output_path, 
        model_path, 
        sdates)
    
    # Write the output JSON file
    with open(output_json_path, "w") as f:
        json.dump(response, f)

    print(response)

    