# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow:2.8.0

RUN apt-get update && apt-get install -y \
    curl \
    jq \
    ffmpeg libsm6 libxext6 gdal-bin \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
COPY segmentation_pipeline.py ./
COPY run.sh ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x run.sh
ENTRYPOINT ["./run.sh"]
# ENTRYPOINT ["ls", "-al"]