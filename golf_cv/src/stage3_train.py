"""
Stage 3 – Model Training
──────────────────────────
• Fine-tunes EfficientNet-B0 (ImageNet pretrained) on the auto-labelled patches
• Weighted sampling to handle class imbalance
• Saves best model to  models/best_model.pth
• Plots training curves and confusion matrix

Run:  python src/stage3_train.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image as PILImage
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BATCH_SIZE, CLASSES, LEARNING_RATE, MODELS_DIR, NUM_CLASSES,
    NUM_EPOCHS, PATCH_SIZE, PATCHES_DIR, PATIENCE,
)

DEVICE = (
    torch.device("cuda")  if torch.cuda.is_available()                else
    torch.device("mps")   if torch.backends.mps.is_available()        else
    torch.device("cpu")
)


class SimulateMowingStripes:
    """Add alternating brightness bands to simulate mowing stripe appearance."""
    def __init__(self, p: float = 0.40):
        self.p = p

    def __call__(self, img: PILImage.Image) -> PILImage.Image:
        if np.random.random() > self.p:
            return img
        arr    = np.array(img).astype(np.float32)
        period = np.random.randint(4, 18)
        factor = np.random.uniform(0.80, 0.94)
        axis   = np.random.randint(0, 2)   # 0 = rows, 1 = cols
        size   = arr.shape[axis]
        for i in range(size):
            if (i // period) % 2 == 0:
                if axis == 0:
                    arr[i] *= factor
                else:
                    arr[:, i] *= factor
        return PILImage.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class SimulateShadow:
    """Darken a random rectangular strip to simulate cast shadow."""
    def __init__(self, p: float = 0.25):
        self.p = p

    def __call__(self, img: PILImage.Image) -> PILImage.Image:
        if np.random.random() > self.p:
            return img
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        factor = np.random.uniform(0.45, 0.72)
        if np.random.random() < 0.5:
            y0 = np.random.randint(0, h // 2)
            y1 = np.random.randint(h // 2, h)
            arr[y0:y1] *= factor
        else:
            x0 = np.random.randint(0, w // 2)
            x1 = np.random.randint(w // 2, w)
            arr[:, x0:x1] *= factor
        return PILImage.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ── dataset ──────────────────────────────────────────────────────────────────

class PatchDataset(Dataset):
    def __init__(self, split: str, transform=None):
        self.transform = transform
        self.samples   = []
        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = PATCHES_DIR / split / cls_name
            if not cls_dir.exists():
                continue
            for p in cls_dir.glob("*.jpg"):
                self.samples.append((str(p), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_weights(self) -> torch.Tensor:
        counts = np.zeros(NUM_CLASSES)
        for _, lbl in self.samples:
            counts[lbl] += 1
        w = 1.0 / (counts + 1)
        return torch.FloatTensor([w[lbl] for _, lbl in self.samples])


# ── model ────────────────────────────────────────────────────────────────────

def build_model() -> nn.Module:
    model       = models.efficientnet_b0(weights="IMAGENET1K_V1")
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model.to(DEVICE)


# ── one epoch ────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if training:
                optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            preds       = out.argmax(1)
            correct    += (preds == labels).sum().item()
            total      += len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ── main ─────────────────────────────────────────────────────────────────────

def train():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(" Stage 3 – Model Training")
    print(f"{'='*55}")
    print(f"  Device : {DEVICE}\n")

    tf_train = T.Compose([
        T.ToPILImage(),
        T.Resize((PATCH_SIZE, PATCH_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        SimulateMowingStripes(p=0.40),
        SimulateShadow(p=0.25),
        T.ColorJitter(brightness=0.40, contrast=0.35, saturation=0.30, hue=0.08),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_val = T.Compose([
        T.ToPILImage(),
        T.Resize((PATCH_SIZE, PATCH_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("  Loading datasets …")
    train_ds = PatchDataset("train", tf_train)
    val_ds   = PatchDataset("val",   tf_val)
    print(f"    Train patches : {len(train_ds)}")
    print(f"    Val   patches : {len(val_ds)}\n")

    if len(train_ds) == 0:
        sys.exit("ERROR: no training patches found – run Stage 2 first.")

    sampler      = WeightedRandomSampler(train_ds.class_weights(), len(train_ds))
    pin_mem      = DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=pin_mem)

    model     = build_model()
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model : EfficientNet-B0  ({n_params:.1f}M parameters)\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    history   = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_acc  = 0.0
    no_improve = 0

    print(f"  Training up to {NUM_EPOCHS} epochs  (patience={PATIENCE}) …\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc, *_ = _run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, val_preds, val_labels = _run_epoch(model, val_loader, criterion)
        scheduler.step()

        for k, v in zip(("train_loss","train_acc","val_loss","val_acc"),
                         (tr_loss, tr_acc, va_loss, va_acc)):
            history[k].append(v)

        tag = ""
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), MODELS_DIR / "best_model.pth")
            no_improve = 0
            tag = "  ← best"
        else:
            no_improve += 1

        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
              f"va_loss={va_loss:.4f}  va_acc={va_acc:.3f}{tag}")

        if no_improve >= PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            break

    print(f"\n  Best val accuracy : {best_acc:.3f}")

    # save history
    with open(MODELS_DIR / "history.json", "w") as f:
        json.dump(history, f)

    # training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train"); ax1.plot(history["val_loss"], label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(history["train_acc"], label="Train"); ax2.plot(history["val_acc"], label="Val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.suptitle("Training History – Golf Course Classifier")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "training_curves.png", dpi=150)
    plt.close()
    print(f"  Curves  → {MODELS_DIR / 'training_curves.png'}")

    # confusion matrix on best model
    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pth", map_location=DEVICE))
    _, _, val_preds, val_labels = _run_epoch(model, val_loader, criterion)
    cm = confusion_matrix(val_labels, val_preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES,
                cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix (val set)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  CM      → {MODELS_DIR / 'confusion_matrix.png'}")

    print("\n  Classification Report:")
    print(classification_report(val_labels, val_preds, target_names=CLASSES))
    print(f"\n✓  Stage 3 complete.  Model → {MODELS_DIR / 'best_model.pth'}\n")


if __name__ == "__main__":
    train()
