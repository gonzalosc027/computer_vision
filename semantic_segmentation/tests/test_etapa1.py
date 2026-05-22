import pytest
import numpy as np
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"


def count_images(split: str) -> int:
    split_dir = DATA_RAW / split
    return len(list(split_dir.glob("img_*.jpg")))


def count_masks(split: str) -> int:
    split_dir = DATA_RAW / split
    return len(list(split_dir.glob("mask_*.png")))


def test_minimum_train_images():
    n = count_images("train")
    assert n >= 2500, f"Se encontraron solo {n} imágenes en train (mínimo 2500)"


def test_minimum_val_images():
    n = count_images("validation")
    assert n >= 400, f"Se encontraron solo {n} imágenes en validation (mínimo 400)"


def test_minimum_test_images():
    n = count_images("test")
    assert n >= 150, f"Se encontraron solo {n} imágenes en test (mínimo 150)"


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_image_mask_pairs_exist(split):
    split_dir = DATA_RAW / split
    img_files = sorted(split_dir.glob("img_*.jpg"))
    assert len(img_files) > 0, f"No hay imágenes en {split}"

    missing = []
    for img_path in img_files[:100]:
        mask_path = split_dir / img_path.name.replace("img_", "mask_").replace(".jpg", ".png")
        if not mask_path.exists():
            missing.append(img_path.name)

    assert len(missing) == 0, f"Máscaras faltantes en {split}: {missing[:5]}"


@pytest.mark.parametrize("split", ["train", "validation"])
def test_mask_values_in_range(split):
    split_dir = DATA_RAW / split
    mask_files = sorted(split_dir.glob("mask_*.png"))[:50]
    assert len(mask_files) > 0, f"No hay máscaras en {split}"

    for mask_path in mask_files:
        mask = np.array(Image.open(mask_path))
        assert mask.min() >= 0, f"Valor negativo en {mask_path.name}"
        assert mask.max() <= 149, f"Valor > 149 ({mask.max()}) en {mask_path.name}"


def test_samples_directory_populated():
    samples_dir = BASE_DIR / "data" / "samples"
    img_count = len(list(samples_dir.glob("img_*.jpg")))
    assert img_count >= 5, f"Solo {img_count} imágenes en data/samples/ (mínimo 5)"


def test_images_are_valid_rgb():
    split_dir = DATA_RAW / "train"
    img_files = sorted(split_dir.glob("img_*.jpg"))[:20]

    for img_path in img_files:
        img = Image.open(img_path)
        assert img.mode == "RGB", f"{img_path.name} no es RGB: {img.mode}"
        assert img.width > 0 and img.height > 0


def test_metadata_file_exists():
    meta_path = DATA_RAW / "metadata.json"
    assert meta_path.exists(), "No se encontró data/raw/metadata.json"

    import json
    with open(meta_path) as f:
        meta = json.load(f)
    assert "splits" in meta
    assert "num_classes" in meta
    assert meta["num_classes"] == 150
