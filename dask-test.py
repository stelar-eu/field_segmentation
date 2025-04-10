# General imports
import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import time
import sys
from tqdm import tqdm
from stelar_spatiotemporal.eolearn.core import EOPatch, OverwritePermission, FeatureType, SaveTask, LoadTask
import geopandas as gpd
from stelar_spatiotemporal.segmentation.segmentation import combine_shapes_recursive
from shapely.ops import unary_union
from shapely.geometry import box

LOCAL_DIR = "/home/jens/ownCloud/Documents/3.Werk/0.TUe_Research/0.STELAR/0.VISTA/VISTA_workbench/data/evaluation/33uup/28-05-2023"


plet_dir = os.path.join(LOCAL_DIR, 'patchlets')
vecs_dir = os.path.join(LOCAL_DIR, 'contours')

# Get all patchlet vector files
vec_paths = glob.glob(os.path.join(vecs_dir, "*.gpkg"))

# Sort the vector paths by the x and y coordinates in the filename
# def sort_key(path):
#     # Extract the x and y coordinates from the filename
#     filename = os.path.basename(path).replace('.gpkg', '')
#     parts = filename.split('_')
#     x = int(parts[1])
#     y = int(parts[2])
#     return (x, y)

# vec_paths = sorted(vec_paths, key=sort_key)

# Read all shapes
dfs = []
for vec_path in tqdm(vec_paths, desc="Reading shapes", unit="file"):
    try:
        df = gpd.read_file(vec_path)

        # Only include the fields that are not empty
        if not df.empty:
            dfs.append(df)
    except Exception as e:
        print(f"Error reading {vec_path}: {e}")

# Create a global df from it
start = time.time()
global_df = pd.concat(dfs)

# Filter all big fields from the shapes
global_df = global_df[global_df.geometry.area < 1e6]

print("Number of fields: ", len(global_df))

# Get the intersection shape from the bounding boxes of the individual patchlets
boxes = [box(*df.geometry.total_bounds) for df in dfs]
intersection_shape = unary_union([box.intersection(other_box) for box in boxes for other_box in boxes if box != other_box])

# Get the intersection of all fields (i.e., keep only the fields in global_df that have some area in common with shape_df)
isct_df = global_df[global_df.intersects(intersection_shape)]

print("Number of fields in intersection: ", len(isct_df))

# Take the union of the fields in the sntersection to make sure we do not have duplicates
isct_df = isct_df.dissolve().explode()

print("Number of fields in intersection after dissolving: ", len(isct_df))

# Get the difference of all fields (i.e., keep only the fields in global_df that do not have any area in common with shape_df)
diff_df = global_df[~global_df.intersects(intersection_shape)]

print("Number of fields in difference: ", len(diff_df))

# Now combine the intersection and difference dataframes to get the final dataframe
final_df = pd.concat([isct_df, diff_df])

print("Number of fields in final dataframe: ", len(final_df))

print("Total time: ", time.time() - start)

# Reset the index of the final dataframe
final_df.reset_index(drop=True, inplace=True)

# Save the final dataframe to a file
outpath = os.path.join(LOCAL_DIR, 'combined_fields.gpkg')
final_df.to_file(outpath)


"""
LOGS:

------------------------------- OLD CODE -----------------------------
---- 10 FILES
Reading shapes
Sorting shapes
18872
Dissolving shapes
Total time:  48.95073175430298
Number of shapes after filtering:  16934
Combining shapes recursively
Combining shapes progress: 30.00%
Total time:  103.28171730041504
Number of shapes after filtering:  16934

---- 20 FILES
Reading shapes
Sorting shapes
35791
Dissolving shapes
Total time:  108.36575651168823
Number of shapes after filtering:  29972
Saving shapes
Combining shapes without parallelization
Total time:  112.14960980415344
Number of shapes after filtering:  29972
Combining shapes recursively
Total time:  282.976702690124500
Number of shapes after filtering:  30124

--- ALL FILES
Reading shapes
Sorting shapes
252162
Combining shapes without parallelization
Dissolving shapes
Total time:  932.3634879589081
Number of shapes after filtering:  200046
Saving shapes
Reading shapes
Sorting shapes
252162
Combining shapes recursively
Total time:  3573.75797629356461%
Number of shapes after filtering:  200712

------------------------------ NEW CODE WITH SEPARATE UNION IN INTERSECTION --------
Reading shapes: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 144/144 [01:52<00:00,  1.28file/s]
Number of fields:  254693
Number of fields in intersection:  85750
Number of fields in intersection after dissolving:  43238
Number of fields in difference:  168943
Number of fields in final dataframe:  212181
Total time:  383.3548962211609
"""
