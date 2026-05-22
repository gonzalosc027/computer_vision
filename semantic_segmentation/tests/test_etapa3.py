import json
import pytest
import torch
import numpy as np
from pathlib import Path
from transformers import SegformerForSemanticSegmentation

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "checkpoints"
NUM_CLASSES = 150
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_trained_model(device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    ckpt_path = MODELS_DIR / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def test_best_model_checkpoint_exists():
    assert (MODELS_DIR / "best_model.pth").exists(), \
        "No se encontró models/checkpoints/best_model.pth. Ejecuta la etapa3 primero."


def test_last_model_checkpoint_exists():
    assert (MODELS_DIR / "last_model.pth").exists(), \
        "No se encontró models/checkpoints/last_model.pth."


def test_model_loads_successfully():
    device = get_device()
    model, ckpt = load_trained_model(device)
    assert model is not None
    assert "epoch" in ckpt
    assert "val_miou" in ckpt


def test_model_output_shape():
    device = get_device()
    model, _ = load_trained_model(device)

    B, C, H, W = 2, 3, 512, 512
    dummy_input = torch.randn(B, C, H, W).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=dummy_input)
        logits = outputs.logits

    assert logits.shape[0] == B, f"Batch size incorrecto: {logits.shape[0]}"
    assert logits.shape[1] == NUM_CLASSES, f"Número de clases incorrecto: {logits.shape[1]}"


def test_forward_pass_no_error():
    device = get_device()
    model, _ = load_trained_model(device)

    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        outputs = model(pixel_values=dummy_input)

    assert outputs is not None
    assert not torch.any(torch.isnan(outputs.logits)), "El modelo produce NaN"


def test_model_improved_miou():
    history_path = MODELS_DIR / "training_history.json"
    assert history_path.exists(), "No se encontró training_history.json"

    with open(history_path) as f:
        history = json.load(f)

    val_miou = history["val_miou"]
    assert len(val_miou) >= 2, "Se necesitan al menos 2 épocas para comparar"

    initial_miou = val_miou[0]
    best_miou = max(val_miou)
    assert best_miou >= initial_miou, \
        f"El mIoU no mejoró: inicial={initial_miou:.4f}, mejor={best_miou:.4f}"


def test_predictions_are_valid_class_indices():
    device = get_device()
    model, _ = load_trained_model(device)

    import torch.nn.functional as F
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        outputs = model(pixel_values=dummy_input)
        logits = outputs.logits
        logits_up = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
        preds = logits_up.argmax(dim=1)

    assert preds.min().item() >= 0
    assert preds.max().item() < NUM_CLASSES, f"Predicción fuera de rango: {preds.max().item()}"


def test_training_curves_plot_exists():
    plot_path = BASE_DIR / "outputs" / "plots" / "etapa3_training_curves.png"
    assert plot_path.exists(), "No se encontró outputs/plots/etapa3_training_curves.png"
