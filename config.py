import os
from datetime import datetime
from pathlib import Path

# Base directory - update this path to match your setup
BASE_DIR = '/Users/shresthshrivastava/Documents/LIDAR_Side_Project/data'

# Data directories
DATA_DIR = BASE_DIR
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')

# Raw data locations
DSM_DIR = os.path.join(BASE_DIR, "scotland-dsm-2m_5938282/nt")
DTM_DIR = os.path.join(BASE_DIR, "scotland-dtm-2m_5938283/nt")
LAZ_DIR = os.path.join(BASE_DIR, "scotland-laz-1_5938284/nt")

# === Timestamped Output Folder (Stable across script imports) ===
TIMESTAMP = os.environ.get("RUN_TIMESTAMP")
if not TIMESTAMP:
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["RUN_TIMESTAMP"] = TIMESTAMP

RUN_NAME = f"Multiple_Output_{TIMESTAMP}"
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, 'output')
RUN_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, RUN_NAME)

# Subfolders for organization
PATHS_DIR = os.path.join(RUN_OUTPUT_DIR, "paths")
VISUALIZATION_DIR = os.path.join(RUN_OUTPUT_DIR, "visualizations")

# Make sure they exist
for directory in [OUTPUT_BASE_DIR, RUN_OUTPUT_DIR, PATHS_DIR, VISUALIZATION_DIR]:
    os.makedirs(directory, exist_ok=True)

# Processed data directory
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MERGED_DTM_PATH = os.path.join(PROCESSED_DATA_DIR, "merged_dtm.tif")
OBSTACLES_PATH = os.path.join(PROCESSED_DATA_DIR, "obstacles.tif")

# Algorithm parameters
OBSTACLE_THRESHOLD = 10  # meters
ENERGY_FACTOR_HORIZONTAL = 1.0
ENERGY_FACTOR_VERTICAL = 2.0
DOWNSAMPLE_FACTOR = 10
MAX_RUNTIME_SECONDS = 180
DIRECT_PATH_BIAS = 0.8

# Path smoothing parameters
MAX_SLOPE_DEG = 30
SMOOTHING_ITERATIONS = 10
