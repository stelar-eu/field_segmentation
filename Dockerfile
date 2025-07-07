# Use an official Python runtime as a parent image
FROM python:3.9.16-slim

RUN apt-get update && apt-get install -y \
    curl \
    jq \
    ffmpeg libsm6 libxext6 gdal-bin \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
COPY main.py ./
COPY run.sh ./
COPY resources/input.json ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x run.sh
CMD [ "python", "main.py", "input.json", "output.json" ]
# ENTRYPOINT ["./run.sh"]
# ENTRYPOINT ["ls", "-al"]