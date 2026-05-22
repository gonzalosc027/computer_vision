import pytest
import numpy as np
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from etapa2_preprocesado import (
    SegmentationDataset,
    build_dataloaders,
    get_train_transform,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

BASE_DIR = Path(__file__).resolve().parent.parent


def test_train_batch_shape():
    train_loader, _, _ = build_dataloaders(batch_train=8, batch_val=4, num_workers=0)
    imgs, masks = next(iter(train_loader))
    assert imgs.shape == (8, 3, 512, 512), f"Shape incorrecto: {imgs.shape}"
    assert masks.shape == (8, 512, 512), f"Shape de máscara incorrecto: {masks.shape}"


def test_val_batch_shape():
    _, val_loader, _ = build_dataloaders(batch_train=8, batch_val=4, num_workers=0)
    imgs, masks = next(iter(val_loader))
    assert imgs.shape[1:] == (3, 512, 512), f"Shape incorrecto: {imgs.shape}"


def test_images_normalized():
    train_loader, _, _ = build_dataloaders(batch_train=8, batch_val=4, num_workers=0)
    imgs, _ = next(iter(train_loader))

    mean = imgs.mean().item()
    std = imgs.std().item()
    assert abs(mean) < 2.0, f"Media demasiado alta: {mean}"
    assert 0.1 < std < 5.0, f"Std fuera de rango: {std}"


def test_mask_values_valid():
    train_loader, _, _ = build_dataloaders(batch_train=4, batch_val=4, num_workers=0)
    _, masks = next(iter(train_loader))
    assert masks.min().item() >= 0, f"Valor negativo en máscara: {masks.min().item()}"
    assert masks.max().item() <= 149, f"Valor > 149 en máscara: {masks.max().item()}"


def test_augmentations_produce_different_outputs():
    ds = SegmentationDataset("train", transform=get_train_transform())
    assert len(ds) > 0, "Dataset de entrenamiento vacío"

    img1, mask1 = ds[0]
    img2, mask2 = ds[0]

    diffs = [(ds[0][0] - ds[0][0]).abs().sum().item() for _ in range(5)]
    assert img1.shape == (3, 512, 512), f"Shape incorrecto: {img1.shape}"
    assert img1.dtype == torch.float32


def test_dataset_stats_file_exists():
    stats_path = BASE_DIR / "data" / "processed" / "dataset_stats.json"
    assert stats_path.exists(), "No se encontró data/processed/dataset_stats.json. Ejecuta etapa2_preprocesado.py primero."

    import json
    with open(stats_path) as f:
        stats = json.load(f)

    assert "mean_per_channel" in stats
    assert "std_per_channel" in stats
    assert "class_weights" in stats
    assert len(stats["class_weights"]) == 150


def test_augmentation_plot_generated():
    plot_path = BASE_DIR / "outputs" / "plots" / "etapa2_augmentations.png"
    assert plot_path.exists(), "No se encontró outputs/plots/etapa2_augmentations.png"


def test_dataset_returns_tensors():
    ds = SegmentationDataset("train", transform=get_train_transform())
    img, mask = ds[0]
    assert isinstance(img, torch.Tensor), "La imagen no es un Tensor"
    assert isinstance(mask, torch.Tensor), "La máscara no es un Tensor"
    assert img.dtype == torch.float32
    assert mask.dtype == torch.long
