# Use an official Python runtime as a parent image
# FROM python:3.9.16-slim
FROM tensorflow/tensorflow:2.15.0-gpu

RUN apt-get update && apt-get install -y \
    curl \
    jq \
    ffmpeg libsm6 libxext6 gdal-bin \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
COPY main.py ./
COPY run.sh ./
COPY src ./src
# COPY resources ./resources

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ENTRYPOINT [ "python", "-u", "main.py", "resources/input.json", "resources/output.json" ]

RUN chmod +x run.sh
ENTRYPOINT ["./run.sh"]
# ENTRYPOINT ["ls", "-al"]