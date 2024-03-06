# Field Segmentation Module
## Description
Field Segmentation module, as part of STELAR Work Package 4 (WP4), Task 4.2: Spatio-temporal data labeling.
This module takes as input a satellite image with reflectance bands B2, B3, B4 (RGB), and B8 (near-infrared), and outputs a segmentation map, as a collection of polygons, where each polygon represents a field. 
The polygons can later be used to mask a satellite image covering an overlapping area, to retrieve the data of pixels that fall within the field boundaries. 
We consider a field or agricultural parcel as a spatially homogeneous land unit used for agricultural purposes, where a single crop type is grown. The underlying assumption here is that boundaries of agricultural parcels are visible and spatially resolvable from remotely sensed imagery.

This module is part of the KLMS Tools in the STELAR architecture, which can be invoked by the workflow execution engines. Due to its benefits with respect to both quality and efficiency of downstream tasks, the module can be used in combination with a
variety of other tools, to form a workflow for several STELAR use cases. For instance, in the context of *crop classification* and *yield prediction*, the field segmentation module can be used to extract the reflectance values of pixels that fall within the boundaries of a field, which can then be used as input for a classification or prediction model. 
In the context of *cloud removal* (discussed under the module of data imputation in Deliverable 3.2 Spatio-temporal data alignment and correlations), field segmentation can be used to reduce the amount of time series to be imputed by aggregating the values of pixels that fall within the boundaries of the field, using the [Data Imputation Module](TODO link to the module).
This is relevant because complex imputation algorithms often depend on the number of time series in the dataset, and reducing this number can therefore lead to significant improvements in efficiency. 
Also note that aggregation of pixel values within a field can be seen as a spatial imputation method in itself; when clouds cover only part of a certain field at a certain time step, the aggregated value of the field can be used as an approximation of
the values of the covered pixels. 
All three of these tasks are relevant for use case *B.1 Yield prediction for agricultural decision making* and *B.2 Food security through Early Warnings*. 
Therefore, field segmentation can play a crucial role in the quality and scalability of the final workflows for these use
cases.

|  |  |
| --- | --- |
| Context in project: | WP4 Task 4.2: Spatio-Temporal Data Labeling |
| Relevant other tools: | [Time Series Extraction Module](https://github.com/stelar-eu/spatiotemporal_timeseries_extraction), [Data Imputation Module](TODO:LINK), [Crop Classification Module](TODO:LINK) |
| Relevant use cases: | B.1 Yield prediction for agricultural decision making, B.2 Food security through Early Warnings |
| Report: | Deliverable 4.1 Spatio-Temporal and Textual Data Annotation |

## Input format
The module takes as input the following:
1. *b2_path* (required): Path to the folder containing the B2 band (blue) of the input images as RAS and RHD files.
2. *b3_path* (required): Path to the folder containing the B3 band (green) of the input images as RAS and RHD files.
3. *b4_path* (required): Path to the folder containing the B4 band (red) of the input images as RAS and RHD files.
4. *b8_path* (required): Path to the folder containing the B8A band (Near-Infrared) of the input images as RAS and RHD files.

The directories of the above paths should have the following structure:
```
    input_dir
    ├── image1.RAS
    ├── image1.RHD
    ├── image2.RAS
    └── image2.RHD
```
RAS files are compressed binary files containing the reflectance values of satellite images. 
Each RAS file should have an accompanying header file (.RHD), which contains the metadata of the RAS file such as the bounding box, the coordinate reference system and the timestamps of the images. 
Based on the header files, the script first checks if the RAS files are aligned, i.e., if they have the same bounding box, coordinate reference system and timestamps. 
If the RAS files are not aligned, the script will raise an error.
**Note**: The input paths can be either a local path or a path to a folder in a MinIO object storage. In the latter case, the path should start with `s3://` followed by the MinIO server address and the bucket name, e.g., `s3://localhost:9000/mybucket/input_dir`. Also, the MinIO access key and secret key should be passed as arguments (see below).

5. *output_path* (required): Path to the folder where the output files will be saved. 
**Note**: The output path can be either a local path or a path to a folder in a MinIO object storage. In the latter case, the path should start with `s3://` followed by the MinIO server address and the bucket name, e.g., `s3://localhost:9000/mybucket/output_dir`. Also, the MinIO access key and secret key should be passed as arguments (see below).

6. *model_path* (required): Path to the folder containing the trained model for field segmentation. The model is a .h5 file, and contains a model following the Res-UNet architecture of Sentinel-Hub [github](https://github.com/sentinel-hub/field-delineation). The model is trained on Sentinel-2 images and is used to segment the fields in the input images.

7. *sdates* (optional): A comma-separated list of dates in the format `YYYY-MM-DD` for which the field segmentation should be performed. These dates should be a subset of the dates in the input images. If this argument is not provided, the field segmentation will be performed for all the dates in the input images.

8. *MINIO_ACCESS_KEY* (optional): Access key of the MinIO server. Required if the input or output paths are in a MinIO object storage.

9. *MINIO_SECRET_KEY* (optional): Secret key of the MinIO server. Required if the input or output paths are in a MinIO object storage.

## Output format
The module outputs a shapefile named `fields.gpkg`  containing the field boundaries as polygons. The shapefile is saved in the output folder specified by the *output_path* argument.

## Metrics
The module outputs the following metrics about the run as metadata:
1. *start_time*: The start time of the run.
2. *end_time*: The end time of the run.
3. *duration*: The duration of the run in seconds.
4. *status*: The status of the run, which can be either "success" or "failure".
5. *error_message*: The error message in case the run failed.
6. *input_shape*: The shape of the input images, in the format (height, width).
7. *sdates*: The dates for which the field segmentation was performed.
8. *model_path*: The path to the trained model used for field segmentation.
9. *n_fields*: The number of fields detected in the input images.

## Installation & Example Usage
The module can be installed either by (1) cloning the repository and building the Docker image, or (2) by pulling the image from DockerHub.
Cloning the repository and building the Docker image:
```bash
git clone https://github.com/stelar-eu/field_segmentation.git
cd field_segmentation
docker build -t alexdarancio7/stelar_field_segmentation:latest .
```
Pulling the image from DockerHub:
```bash
docker pull alexdarancio7/stelar_field_segmentation:latest
```
### Example Usage
Then, given we have the following input parameters:
- *b2_path*: `s3://localhost:9000/path/to/b2_input_dir`
- *b3_path*: `s3://localhost:9000/path/to/b3_input_dir`
- *b4_path*: `s3://localhost:9000/path/to/b4_input_dir`
- *b8_path*: `s3://localhost:9000/path/to/b8_input_dir`
- *output_path*: `s3://localhost:9000/path/to/output_dir`
- *model_path*: `s3://localhost:9000/path/to/model_dir`
- *sdates*: `2021-01-01,2021-01-02`
- *MINIO_ACCESS_KEY*: `minio`
- *MINIO_SECRET_KEY*: `minio123`

We can run the module as follows:
```bash
docker run -it \
--network="host" \
alexdarancio7/stelar_field_segmentation \
--b2_path s3://localhost:9000/path/to/b2_input_dir \
--b3_path s3://localhost:9000/path/to/b3_input_dir \
--b4_path s3://localhost:9000/path/to/b4_input_dir \
--b8_path s3://localhost:9000/path/to/b8_input_dir \
--output_path s3://localhost:9000/path/to/output_dir \
--model_path s3://localhost:9000/path/to/model_dir \
--sdates '2021-01-01,2021-01-02' \
--MINIO_ACCESS_KEY minio \
--MINIO_SECRET_KEY minio123
```

## License & Acknowledgements
This module is part of the STELAR project, which is funded by the European Union’s Europe research and innovation programme under grant agreement No 101070122.
The module is licensed under the MIT License (see [LICENSE](LICENSE) for details).