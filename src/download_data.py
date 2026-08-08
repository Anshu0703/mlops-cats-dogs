"""
Downloads the Cats and Dogs Classification Dataset from Kaggle
into data/raw/. Requires a Kaggle API token at ~/.kaggle/kaggle.json.

Usage:
    python src/download_data.py
"""
import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "bhavikjikadara/dog-and-cat-classification-dataset"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/kaggle.json

    print(f"Downloading '{DATASET}' into {RAW_DIR} ...")
    api.dataset_download_files(DATASET, path=str(RAW_DIR), unzip=True)
    print("Download complete.")


if __name__ == "__main__":
    download_dataset()