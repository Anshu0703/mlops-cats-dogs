"""
Unit tests for data preprocessing functions.
"""
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocess import is_valid_image


def test_is_valid_image_accepts_real_image(tmp_path):
    """A properly saved JPEG should be recognized as valid."""
    img_path = tmp_path / "valid.jpg"
    Image.new("RGB", (50, 50), color="red").save(img_path)

    assert is_valid_image(img_path) is True


def test_is_valid_image_rejects_corrupt_file(tmp_path):
    """A file with .jpg extension but garbage bytes should be rejected."""
    bad_path = tmp_path / "corrupt.jpg"
    bad_path.write_bytes(b"this is not a real image file")

    assert is_valid_image(bad_path) is False


def test_is_valid_image_rejects_empty_file(tmp_path):
    """A 0-byte file (a known issue in this dataset) should be rejected."""
    empty_path = tmp_path / "empty.jpg"
    empty_path.write_bytes(b"")

    assert is_valid_image(empty_path) is False