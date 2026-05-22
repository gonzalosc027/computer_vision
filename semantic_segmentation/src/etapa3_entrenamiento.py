import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from etapa2_preprocesado import build_dataloaders

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "checkpoints"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

NUM_CLASSES = 150
EPOCHS = 10
LR = 6e-5
WEIGHT_DECAY = 0.01
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_class_weights(device):
    stats_path = DATA_PROCESSED / "dataset_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        weights = torch.tensor(stats["class_weights"], dtype=torch.float32).to(device)
    else:
        weights = torch.ones(NUM_CLASSES, dtype=torch.float32).to(device)
    return weights


def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = NUM_CLASSES) -> float:
    iou_list = []
    preds = preds.view(-1)
    targets = targets.view(-1)

    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union > 0:
            iou_list.append(intersection / union)

    return float(np.mean(iou_list)) if iou_list else 0.0


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [train]", leave=False)
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(pixel_values=imgs)
        logits = outputs.logits

        logits_up = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        B, C, H, W = logits_up.shape
        logits_flat = logits_up.permute(0, 2, 3, 1).reshape(-1, C)
        masks_flat = masks.reshape(-1)

        loss = criterion(logits_flat, masks_flat)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    for imgs, masks in tqdm(loader, desc="[val]", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(pixel_values=imgs)
        logits = outputs.logits
        logits_up = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        B, C, H, W = logits_up.shape
        logits_flat = logits_up.permute(0, 2, 3, 1).reshape(-1, C)
        masks_flat = masks.reshape(-1)

        loss = criterion(logits_flat, masks_flat)
        total_loss += loss.item()
        n_batches += 1

        preds = logits_up.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    preds_cat = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    miou = compute_miou(preds_cat, targets_cat)

    return total_loss / max(n_batches, 1), miou


def save_training_curves(history: dict):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Etapa 3 — Curvas de Entrenamiento SegFormer-B0", fontsize=14, fontweight='bold')

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss vs Val Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["val_miou"], "g-o", label="Val mIoU")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("mIoU")
    ax2.set_title("mIoU en Validación")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    out_path = PLOTS_DIR / "etapa3_training_curves.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\nCurvas guardadas en {out_path}")


def print_epoch_table(history: dict):
    print("\n" + "=" * 65)
    print(f"{'Época':>6} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val mIoU':>8} | {'Tiempo':>8}")
    print("-" * 65)
    for i, (tl, vl, mi, t) in enumerate(zip(
        history["train_loss"], history["val_loss"],
        history["val_miou"], history["epoch_time"]
    )):
        print(f"{i+1:>6} | {tl:>10.4f} | {vl:>8.4f} | {mi:>8.4f} | {t:>6.1f}s")
    print("=" * 65)
    best_epoch = int(np.argmax(history["val_miou"])) + 1
    best_miou = max(history["val_miou"])
    print(f"\nMejor mIoU: {best_miou:.4f} en época {best_epoch}")


def main():
    print("=" * 60)
    print("ETAPA 3: Entrenamiento SegFormer-B0")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"\nDispositivo: {device}")

    print(f"\nCargando modelo {MODEL_NAME}...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    class_weights = load_class_weights(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print("\nCargando dataloaders...")
    train_loader, val_loader, _ = build_dataloaders(batch_train=8, batch_val=4, num_workers=2)
    print(f"  Train: {len(train_loader.dataset)} imgs  |  Val: {len(val_loader.dataset)} imgs")

    history = {"train_loss": [], "val_loss": [], "val_miou": [], "epoch_time": []}
    best_miou = 0.0
    initial_miou = None

    print(f"\nIniciando entrenamiento por {EPOCHS} épocas...\n")

    for epoch in range(EPOCHS):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_miou = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)
        history["epoch_time"].append(elapsed)

        if initial_miou is None:
            initial_miou = val_miou

        print(f"Época {epoch+1:>2}/{EPOCHS}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_mIoU={val_miou:.4f}  t={elapsed:.1f}s")

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_miou": val_miou,
            "val_loss": val_loss,
            "history": history,
        }, MODELS_DIR / "last_model.pth")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": val_miou,
                "val_loss": val_loss,
                "history": history,
            }, MODELS_DIR / "best_model.pth")
            print(f"  -> Mejor modelo guardado (mIoU={best_miou:.4f})")

    print_epoch_table(history)
    save_training_curves(history)

    with open(MODELS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nmIoU inicial: {initial_miou:.4f}  ->  Mejor mIoU: {best_miou:.4f}")
    if best_miou > initial_miou:
        print("El modelo ha aprendido correctamente.")
    print("\nEtapa 3 completada.")


if __name__ == "__main__":
    main()
