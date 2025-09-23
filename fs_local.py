from stelar.client import Client, TaskSpec
from datetime import datetime
import subprocess

BASE_URL = "https://klms.stelar.gr"
USERNAME = "jdhondt"
PASSWORD = "1234"

# Replace with the tile ID you want to run FS on
TILE = "33UUP"
YEAR = "2022"

def info(message: str):
    print(f"[INFO] {message}")


def error(message: str):
    print(f"[ERROR] {message}")


# Initialize the client with the base URL and credentials
c = Client(base_url=BASE_URL, username=USERNAME, password=PASSWORD)
info(f"Connected to STELAR KLMS @ {c._base_url} as {c._username}")
info(f"Running FS on tile {TILE}")

# --------------------------------------------------------------------------------------------
# Find the dataset with the reflectance files for the Field Segmentation (FS) task
# --------------------------------------------------------------------------------------------

# Find the dataset with the reflectance files for the given tile and year
try:
    reflectance_files = c.datasets[f"{TILE.lower()}-sentinel2-{YEAR}-tifs"]
    info(
        f"Browse selected dataset at: {c._base_url}/console/v1/catalog/{reflectance_files.id}"
    )
except Exception as e:
    error(f"Dataset {TILE.lower()}-sentinel2-{YEAR}-tifs not found: {e}. Interrupting.")
    raise

# --------------------------------------------------------------------------------------------
# Find/Create the workflow process for the Field Segmentation (FS) task
# --------------------------------------------------------------------------------------------

# Create or use a workflow process to run the FS task in, named after the tile and year
try:
    process_name = "ucb2-pipeline-" + TILE.lower() + "-" + YEAR
    p = c.processes.create(
        name=process_name,
        title=f"UCB2 Pipeline on {TILE.upper()} - {YEAR}",
    )
except Exception:
    p = c.processes["ucb2-pipeline-" + TILE.lower() + "-" + YEAR]

info(f"Using process {p.name}. Browse it at: {c._base_url}/console/v1/process/{p.id}")


# -------------------------------------------------------------------------------------------
# Building the Task Specification for Field Segmentation (FS)
# -------------------------------------------------------------------------------------------

# Create a task specification for the FS task
# Use the current timestamp to create a unique names for files and outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Begin with an empty TaskSpec
field_segm = TaskSpec(name=f"Field Segmentation {TILE} during {YEAR}")

# Get the reflectance file from the selected dataset
r_file = reflectance_files.resources[0]


# Define the RGB input of Field Segmentation. The IC.TIF goes here.
field_segm.i(RGB=r_file)

# Provide some local dataset aliases to use them to store the output
# In this case we are gonna store the outputs in the same dataset as the inputs
field_segm.d(alias="fsegm", dset=reflectance_files)

# Set the parameters of the executions
model_path = c.datasets["resunet-model-0-fold"].resources[0].url

field_segm.p(model_path=model_path, sdates="2023-06-12")

# Define the outputs urls and metadata handling
map_url = (
    f"s3://vista-bucket/Pilot_B/UCB2/{TILE.upper()}/proc-{p.id}/fields_{timestamp}.gpkg"
)
metadata_url = (
    f"s3://vista-bucket/Pilot_B/UCB2/{TILE.upper()}/proc-{p.id}/fields_{timestamp}.json"
)

# Set the outputs in the spec
field_segm.o(
    segmentation_map={
        "url": map_url,
        "resource": {"name": "Segmented Fields", "relation": "fields"},
        "dataset": "fsegm",
    }
)
field_segm.o(
    metadata={
        "url": metadata_url,
        "resource": {"name": "Segmented Map Metadata", "relation": "metadata"},
        "dataset": "fsegm",
    }
)


# -------------------------------------------------------------------------------------------
# Register the Field Segmentation task in the selected wf process
# -------------------------------------------------------------------------------------------
fs_task = p.run(field_segm)

info(
    f"Registered Field Segmentation task. Browse it at: {c._base_url}/console/v1/task/{p.id}/{fs_task.id}"
)

# -------------------------------------------------------------------------------------------
# Run the Field Segmentation task using Docker locally. The image is fetched from the STELAR registry
# !!! Important: Make sure you have logged in to the STELAR registry before fetching the image.
# -------------------------------------------------------------------------------------------
image_name = "img.stelar.gr/stelar/field-segmentation:latest"

# Get Task ID and signature
fs_id = str(fs_task.id)
fs_signature = str(fs_task.signature)

docker_command = [
    "docker",
    "run",
    image_name,
    "Bearer " + c.token,
    c.api_url.rstrip("/api/"),
    fs_id,
    fs_signature,
]

# Run the command using subprocess
try:
    process = subprocess.Popen(docker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    if process.returncode != 0:
        print("Error:", process.stderr.read())
except subprocess.CalledProcessError as e:
    print("Error:", e.stderr)
