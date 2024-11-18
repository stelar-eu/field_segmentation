#!/bin/bash

MINIO_ACCESS_KEY="minioadmin"
MINIO_SECRET_KEY="minioadmin"
MINIO_ENDPOINT_URL="http://localhost:9000"

b2_path="s3://stelar-spatiotemporal/RGB_small/B2"
b3_path="s3://stelar-spatiotemporal/RGB_small/B3"
b4_path="s3://stelar-spatiotemporal/RGB_small/B4"
b8_path="s3://stelar-spatiotemporal/RGB_small/B8"
out_path="s3://stelar-spatiotemporal/fields_test.gpkg"
model_path="s3://stelar-spatiotemporal/resunet-a_fold-0_2023-03-27-09-29-38"
sdates="2020-07-27"

python3 segmentation_pipeline.py \
-b2 $b2_path \
-b3 $b3_path \
-b4 $b4_path \
-b8 $b8_path \
-o $out_path \
-m $model_path \
-s $sdates \
--MINIO_ACCESS_KEY $MINIO_ACCESS_KEY \
--MINIO_SECRET_KEY $MINIO_SECRET_KEY \
--MINIO_ENDPOINT_URL $MINIO_ENDPOINT_URL