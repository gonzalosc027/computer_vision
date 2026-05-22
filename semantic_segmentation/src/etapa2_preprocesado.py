import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 150
IMG_SIZE = 512


def get_train_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomRotate90(p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_test_transform():
    return get_val_transform()


class SegmentationDataset(Dataset):
    def __init__(self, split: str, transform=None):
        self.split_dir = DATA_RAW / split
        self.transform = transform

        self.img_paths = sorted(self.split_dir.glob("img_*.jpg"))
        self.mask_paths = [
            self.split_dir / p.name.replace("img_", "mask_").replace(".jpg", ".png")
            for p in self.img_paths
        ]
        pairs = [(i, m) for i, m in zip(self.img_paths, self.mask_paths) if i.exists() and m.exists()]
        self.img_paths, self.mask_paths = zip(*pairs) if pairs else ([], [])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]))

        mask = np.clip(mask, 0, NUM_CLASSES - 1).astype(np.int64)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            aug_mask = augmented["mask"]
            mask = aug_mask.detach().clone().to(dtype=torch.long) if isinstance(aug_mask, torch.Tensor) else torch.tensor(aug_mask, dtype=torch.long)
        else:
            img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            mask = torch.tensor(mask, dtype=torch.long)

        return img, mask


def build_dataloaders(batch_train=8, batch_val=4, num_workers=2):
    train_ds = SegmentationDataset("train", transform=get_train_transform())
    val_ds = SegmentationDataset("validation", transform=get_val_transform())
    test_ds = SegmentationDataset("test", transform=get_test_transform())

    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True,
                              num_workers=num_workers, pin_memory=use_pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_val, shuffle=False,
                            num_workers=num_workers, pin_memory=use_pin)
    test_loader = DataLoader(test_ds, batch_size=batch_val, shuffle=False,
                             num_workers=num_workers, pin_memory=use_pin)

    return train_loader, val_loader, test_loader


def compute_dataset_stats(max_imgs=500):
    print("\nCalculando estadísticas del dataset...")
    ds = SegmentationDataset("train", transform=A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        ToTensorV2(),
    ]))

    n = min(max_imgs, len(ds))
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    class_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for i in tqdm(range(n), desc="Stats"):
        img, mask = ds[i]
        img_np = img.numpy().astype(np.float64) / 255.0
        for c in range(3):
            pixel_sum[c] += img_np[c].sum()
            pixel_sq_sum[c] += (img_np[c] ** 2).sum()
        pixel_count += IMG_SIZE * IMG_SIZE

        unique, counts = np.unique(mask.numpy(), return_counts=True)
        for cls, cnt in zip(unique, counts):
            if 0 <= cls < NUM_CLASSES:
                class_pixel_counts[cls] += cnt

    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)

    total_pixels = class_pixel_counts.sum()
    class_freq = class_pixel_counts / (total_pixels + 1e-8)
    class_weights = np.where(class_freq > 0, 1.0 / (class_freq + 1e-6), 0.0)
    class_weights = np.clip(class_weights, 0, 10.0)
    class_weights = class_weights / class_weights[class_weights > 0].mean()

    return mean.tolist(), std.tolist(), class_pixel_counts.tolist(), class_weights.tolist()


def save_dataset_stats():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    mean, std, class_pixels, class_weights = compute_dataset_stats()

    stats = {
        "mean_per_channel": mean,
        "std_per_channel": std,
        "pixels_per_class": class_pixels,
        "class_weights": class_weights,
    }
    out_path = DATA_PROCESSED / "dataset_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Estadísticas guardadas en {out_path}")
    return stats


def generate_augmentation_plot():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_augment = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        A.RandomRotate90(p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.HueSaturationValue(p=1.0),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=1.0),
    ])

    img_files = sorted((DATA_RAW / "train").glob("img_*.jpg"))[:1]
    if not img_files:
        print("No hay imágenes de entrenamiento. Saltando plot de augmentations.")
        return

    img = np.array(Image.open(img_files[0]).convert("RGB"))

    augments = [
        ("Original", A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])),
        ("HFlip + Brillo", A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=1.0), A.RandomBrightnessContrast(p=1.0)])),
        ("Rotate90", A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.RandomRotate90(p=1.0)])),
        ("GaussianBlur", A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.GaussianBlur(blur_limit=(5, 9), p=1.0)])),
        ("HueSaturation", A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HueSaturationValue(hue_shift_limit=30, p=1.0)])),
        ("ShiftScaleRotate", A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.25, rotate_limit=45, p=1.0)])),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Etapa 2 — Ejemplos de Aumentación de Datos", fontsize=14, fontweight='bold')

    for ax, (name, transform) in zip(axes.flat, augments):
        result = transform(image=img)["image"]
        ax.imshow(result)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out_path = PLOTS_DIR / "etapa2_augmentations.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada en {out_path}")


def main():
    print("=" * 60)
    print("ETAPA 2: Preprocesado y transformaciones")
    print("=" * 60)

    train_loader, val_loader, test_loader = build_dataloaders()

    print(f"\nDataLoaders creados:")
    print(f"  Train  : {len(train_loader.dataset)} imágenes, batch={train_loader.batch_size}")
    print(f"  Val    : {len(val_loader.dataset)} imágenes, batch={val_loader.batch_size}")
    print(f"  Test   : {len(test_loader.dataset)} imágenes, batch={test_loader.batch_size}")

    imgs, masks = next(iter(train_loader))
    print(f"\nShape batch train -> imágenes: {imgs.shape}, máscaras: {masks.shape}")

    stats = save_dataset_stats()
    print(f"\nMedia por canal   : {[f'{v:.4f}' for v in stats['mean_per_channel']]}")
    print(f"Std por canal     : {[f'{v:.4f}' for v in stats['std_per_channel']]}")

    generate_augmentation_plot()
    print("\nEtapa 2 completada.")


if __name__ == "__main__":
    main()
