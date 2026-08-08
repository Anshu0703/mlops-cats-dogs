"""
Preprocesses the raw Cats vs Dogs images:
- Filters out corrupt/unreadable files (this dataset is known to have a few)
- Resizes to 224x224 RGB
- Splits into train/val/test (80/10/10)
- Saves into data/processed/{train,val,test}/{cat,dog}/

Usage:
    python src/preprocess.py
"""
import random
from pathlib import Path
from PIL import Image, UnidentifiedImageError

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "PetImages"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
IMG_SIZE = (224, 224)
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42


def is_valid_image(path: Path) -> bool:
    """Some files in this dataset are corrupt (0-byte or truncated JPEGs)."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def load_valid_filepaths(class_dir: Path) -> list[Path]:
    files = [f for f in class_dir.glob("*.jpg")]
    valid = [f for f in files if is_valid_image(f)]
    print(f"{class_dir.name}: {len(valid)}/{len(files)} valid images")
    return valid


def split_files(files: list[Path]) -> dict[str, list[Path]]:
    random.Random(SEED).shuffle(files)
    n = len(files)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def process_and_save(files: list[Path], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        try:
            with Image.open(f) as img:
                img = img.convert("RGB").resize(IMG_SIZE)
                img.save(out_dir / f.name, "JPEG")
        except (UnidentifiedImageError, OSError):
            continue  # skip any file that fails at save-time too


def main():
    for class_name, label in [("Cat", "cat"), ("Dog", "dog")]:
        class_dir = RAW_DIR / class_name
        valid_files = load_valid_filepaths(class_dir)
        splits = split_files(valid_files)

        for split_name, split_files_list in splits.items():
            out_dir = PROCESSED_DIR / split_name / label
            process_and_save(split_files_list, out_dir)
            print(f"  {split_name}/{label}: {len(split_files_list)} images -> {out_dir}")

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()