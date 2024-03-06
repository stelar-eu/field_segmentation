#!/bin/bash

docker pull alexdarancio7/stelar_field_segmentation:latest

MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_ENDPOINT_URL=localhost:9000

b2_path="s3://$MINIO_ENDPOINT_URL/stelar-spatiotemporal/RGB/B2"
b3_path="s3://$MINIO_ENDPOINT_URL/stelar-spatiotemporal/RGB/B2"
b4_path="s3://$MINIO_ENDPOINT_URL/stelar-spatiotemporal/RGB/B2"
b8_path="s3://$MINIO_ENDPOINT_URL/stelar-spatiotemporal/RGB/B2"
output_path="s3://$MINIO_ENDPOINT_URL/stelar-spatiotemporal/"
model_path="s3://$MINIO_ENDPOINT_URL/stelar-spatiotemporal/resunet-a_avg_2023-03-25-21-24-38"
sdates="2020-07-04,2020-07-07"

docker run -it \
--network="host" \
alexdarancio7/stelar_field_segmentation \
--b2_path $b2_path \
--b3_path $b3_path \
--b4_path $b4_path \
--b8_path $b8_path \
--output_path $output_path \
--model_path $model_path \
--sdates $sdates \
--MINIO_ACCESS_KEY $MINIO_ACCESS_KEY \
--MINIO_SECRET_KEY $MINIO_SECRET_KEY 