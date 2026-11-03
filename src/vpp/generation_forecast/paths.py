"""Canonical repository paths used by the generation forecast pipeline."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPOSITORY_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
RAW_GENERATION_DIR = DATA_DIR / "raw" / "generation"
RAW_GFS_DIR = DATA_DIR / "raw" / "gfs_history_core"
PROCESSED_DIR = DATA_DIR / "processed" / "generation_forecast"
SAMPLE_DIR = DATA_DIR / "sample" / "generation_forecast"
MODEL_DIR = REPOSITORY_ROOT / "artifacts" / "models"
PREDICTION_DIR = REPOSITORY_ROOT / "artifacts" / "predictions"
RESULTS_DIR = REPOSITORY_ROOT / "docs" / "results"

PLANT_MASTER_PATH = REFERENCE_DIR / "plant_location_master_21.csv"
GFS_GRID_MASTER_PATH = REFERENCE_DIR / "jeju_gfs_grids_6.csv"
