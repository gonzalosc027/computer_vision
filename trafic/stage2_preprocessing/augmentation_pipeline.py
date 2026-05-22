import random
import sys
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRAIN_IMGS = ROOT / "data" / "processed" / "train" / "images"
TRAIN_LBLS = ROOT / "data" / "processed" / "train" / "labels"
AUG_IMGS = ROOT / "data" / "augmented" / "images"
AUG_LBLS = ROOT / "data" / "augmented" / "labels"
OUT_DIR = ROOT / "results" / "stage2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline de aumentación
TRANSFORM = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.RandomShadow(p=0.2),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.3),
        A.Perspective(scale=(0.02, 0.05), p=0.2),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,
    ),
)


def load_yolo_labels(label_path: Path) -> tuple[list[int], list[list[float]]]:
    """Carga labels YOLOv8 → (class_ids, bboxes)."""
    class_ids, bboxes = [], []
    if not label_path.exists():
        return class_ids, bboxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_ids.append(int(parts[0]))
        bboxes.append([float(p) for p in parts[1:]])
    return class_ids, bboxes


def save_yolo_labels(label_path: Path, class_ids: list[int], bboxes: list) -> None:
    lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls, (cx, cy, w, h) in zip(class_ids, bboxes)]
    label_path.write_text("\n".join(lines))


def augment_image(img_path: Path, lbl_path: Path, out_img: Path, out_lbl: Path) -> bool:
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    class_ids, bboxes = load_yolo_labels(lbl_path)

    try:
        result = TRANSFORM(image=img_rgb, bboxes=bboxes, class_labels=class_ids)
    except Exception:
        return False

    aug_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_img), aug_img)
    save_yolo_labels(out_lbl, result["class_labels"], result["bboxes"])
    return True


def run_augmentation() -> int:
    AUG_IMGS.mkdir(parents=True, exist_ok=True)
    AUG_LBLS.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(TRAIN_IMGS.glob("*.jpg"))
    if not img_paths:
        print(f"  No hay imágenes en {TRAIN_IMGS}. Ejecuta primero convert_annotations.py")
        return 0

    success = 0
    print(f"  Aumentando {len(img_paths)} imágenes de entrenamiento …")
    for img_path in tqdm(img_paths, unit="img"):
        lbl_path = TRAIN_LBLS / (img_path.stem + ".txt")
        out_img = AUG_IMGS / ("aug_" + img_path.name)
        out_lbl = AUG_LBLS / ("aug_" + img_path.stem + ".txt")

        if out_img.exists():
            success += 1
            continue
        if augment_image(img_path, lbl_path, out_img, out_lbl):
            success += 1

    return success


def demo_augmentation_grid(n_samples: int = 6) -> None:
    """Muestra una cuadrícula Original vs Aumentado para verificación visual."""
    img_paths = sorted(TRAIN_IMGS.glob("*.jpg"))
    if not img_paths:
        return

    random.seed(7)
    samples = random.sample(img_paths, min(n_samples, len(img_paths)))
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 3 * n_samples))
    fig.patch.set_facecolor("#16213e")

    for row, img_path in enumerate(samples):
        lbl_path = TRAIN_LBLS / (img_path.stem + ".txt")
        img_rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        class_ids, bboxes = load_yolo_labels(lbl_path)

        try:
            result = TRANSFORM(image=img_rgb.copy(), bboxes=bboxes, class_labels=class_ids)
            aug_rgb = result["image"]
        except Exception:
            aug_rgb = img_rgb.copy()

        for col, (img, title) in enumerate([(img_rgb, "Original"), (aug_rgb, "Aumentado")]):
            ax = axes[row][col]
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(title, color="white", fontsize=12, fontweight="bold")

    plt.suptitle("Pipeline de aumentación de datos", color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "augmentation_demo.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Demo de aumentación guardado: {out}")


def main() -> None:
    print("Transformaciones activas:")
    for t in TRANSFORM.transforms:
        print(f"  - {type(t).__name__}")

    n_aug = run_augmentation()
    original_count = len(list(TRAIN_IMGS.glob("*.jpg")))
    print(f"\n  Imágenes originales : {original_count}")
    print(f"  Imágenes aumentadas  : {n_aug}")
    print(f"  Total combinado     : {original_count + n_aug}")

    demo_augmentation_grid()
    print("Etapa 2 (aumentación) completada.")


if __name__ == "__main__":
    main()
