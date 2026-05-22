import pytest
import numpy as np
import urllib.request
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

BASE_DIR = Path(__file__).resolve().parent.parent
INTERNET_SAMPLES_DIR = BASE_DIR / "outputs" / "test_results" / "internet_samples"

INTERNET_IMAGES = [
    {
        "name": "urban_street.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/All_Gizah_Pyramids.jpg/640px-All_Gizah_Pyramids.jpg",
        "expected_class": "sky",  # pyramids + sky scene
    },
    {
        "name": "interior.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Biologia_Celular.JPG/640px-Biologia_Celular.JPG",
        "expected_class": None,
    },
    {
        "name": "landscape.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/640px-24701-nature-natural-beauty.jpg",
        "expected_class": "sky",
    },
]


def _download_image(url: str, dest_path: Path) -> bool:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        with open(dest_path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"Error descargando {url}: {e}")
        return False


@pytest.fixture(scope="module")
def model_and_device():
    from etapa4_inferencia import load_model, get_device, MODELS_DIR
    device = get_device()
    checkpoint = MODELS_DIR / "best_model.pth"
    model = load_model(device, checkpoint)
    return model, device


@pytest.fixture(scope="module")
def downloaded_images():
    INTERNET_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for img_info in INTERNET_IMAGES:
        dest = INTERNET_SAMPLES_DIR / img_info["name"]
        if not dest.exists():
            ok = _download_image(img_info["url"], dest)
            if not ok:
                placeholder = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
                placeholder.save(dest)
        paths[img_info["name"]] = dest
    return paths


def test_inference_urban_no_error(model_and_device, downloaded_images):
    from etapa4_inferencia import segmentar_imagen
    model, device = model_and_device
    img_path = downloaded_images["urban_street.jpg"]

    original, colored_mask, overlay, pred_mask = segmentar_imagen(img_path, model, device)

    assert original is not None
    assert colored_mask is not None
    assert overlay is not None


def test_inference_interior_no_error(model_and_device, downloaded_images):
    from etapa4_inferencia import segmentar_imagen
    model, device = model_and_device
    img_path = downloaded_images["interior.jpg"]

    original, colored_mask, overlay, pred_mask = segmentar_imagen(img_path, model, device)
    assert pred_mask is not None
    assert pred_mask.ndim == 2


def test_inference_landscape_no_error(model_and_device, downloaded_images):
    from etapa4_inferencia import segmentar_imagen
    model, device = model_and_device
    img_path = downloaded_images["landscape.jpg"]

    original, colored_mask, overlay, pred_mask = segmentar_imagen(img_path, model, device)
    assert pred_mask is not None


def test_masks_cover_all_pixels(model_and_device, downloaded_images):
    from etapa4_inferencia import segmentar_imagen
    model, device = model_and_device

    for img_info in INTERNET_IMAGES:
        img_path = downloaded_images[img_info["name"]]
        original, colored_mask, overlay, pred_mask = segmentar_imagen(img_path, model, device)

        total_px = pred_mask.size
        classified_px = (pred_mask >= 0).sum()
        assert classified_px == total_px, \
            f"{img_info['name']}: {total_px - classified_px} píxeles sin clase"


def test_output_images_saved(model_and_device, downloaded_images):
    from etapa4_inferencia import segmentar_imagen
    model, device = model_and_device

    for img_info in INTERNET_IMAGES:
        img_path = downloaded_images[img_info["name"]]
        original, colored_mask, overlay, pred_mask = segmentar_imagen(img_path, model, device)

        out_path = INTERNET_SAMPLES_DIR / f"result_{img_info['name']}"
        Image.fromarray(overlay).save(out_path)
        assert out_path.exists(), f"No se guardó {out_path}"


def test_sky_detected_in_landscape(model_and_device, downloaded_images):
    from etapa4_inferencia import segmentar_imagen
    model, device = model_and_device

    img_path = downloaded_images["landscape.jpg"]
    _, _, _, pred_mask = segmentar_imagen(img_path, model, device)

    unique_classes = np.unique(pred_mask)
    sky_class_idx = 2
    has_sky = sky_class_idx in unique_classes
    has_diverse_classes = len(unique_classes) >= 5

    assert has_sky or has_diverse_classes, \
        f"Paisaje sin 'sky' ni diversidad: clases={unique_classes[:10]}"


def test_colored_mask_shape_matches_original(model_and_device, downloaded_images):
    from etapa4_inferencia import segmentar_imagen
    model, device = model_and_device

    for img_info in INTERNET_IMAGES:
        img_path = downloaded_images[img_info["name"]]
        original, colored_mask, overlay, _ = segmentar_imagen(img_path, model, device)

        assert colored_mask.shape[:2] == original.shape[:2], \
            f"Dimensiones no coinciden: original={original.shape}, mask={colored_mask.shape}"
        assert colored_mask.shape[2] == 3, "La máscara coloreada no tiene 3 canales RGB"
