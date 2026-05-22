import os
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_SAMPLES = BASE_DIR / "data" / "samples"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"

TRAIN_SIZE = 3000
VAL_SIZE = 500
TEST_SIZE = 200
SAMPLE_SIZE = 10

ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane",
    "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain",
    "plant", "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house",
    "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock",
    "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column",
    "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table",
    "pillow", "screen door", "stairway", "river", "bridge", "bookcase", "blind",
    "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop", "stove",
    "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine",
    "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning",
    "streetlight", "booth", "television receiver", "airplane", "dirt track", "apparel",
    "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet", "poster",
    "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer",
    "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall", "tent",
    "bag", "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade name",
    "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen",
    "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray",
    "ashcan", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board",
    "shower", "radiator", "glass", "clock", "flag"
]


def save_split(dataset_split, split_name: str, max_items: int, offset: int = 0):
    split_dir = DATA_RAW / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    items = list(dataset_split)
    items = items[:max_items]

    print(f"\nGuardando split '{split_name}' ({len(items)} imágenes)...")
    for i, item in enumerate(tqdm(items, desc=split_name)):
        idx = offset + i
        img_path = split_dir / f"img_{idx:06d}.jpg"
        mask_path = split_dir / f"mask_{idx:06d}.png"

        img = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(img_path, "JPEG", quality=95)

        mask = item["annotation"]
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.array(mask, dtype=np.uint8))
        mask.save(mask_path, "PNG")

    return len(items)


def copy_samples(split_name: str = "train"):
    DATA_SAMPLES.mkdir(parents=True, exist_ok=True)
    split_dir = DATA_RAW / split_name

    img_files = sorted(split_dir.glob("img_*.jpg"))[:SAMPLE_SIZE]
    for img_file in img_files:
        mask_file = split_dir / img_file.name.replace("img_", "mask_").replace(".jpg", ".png")
        shutil.copy2(img_file, DATA_SAMPLES / img_file.name)
        if mask_file.exists():
            shutil.copy2(mask_file, DATA_SAMPLES / mask_file.name)

    print(f"\nCopiadas {len(img_files)} imágenes a data/samples/")


def compute_stats(dataset_split, max_items: int = 500):
    class_counts = np.zeros(150, dtype=np.int64)
    sizes = []

    items = list(dataset_split)[:max_items]
    for item in tqdm(items, desc="Calculando estadísticas"):
        mask = np.array(item["annotation"])
        img = item["image"]
        sizes.append(img.size)

        unique, counts = np.unique(mask, return_counts=True)
        for cls, cnt in zip(unique, counts):
            if cls < 150:
                class_counts[cls] += cnt

    return class_counts, sizes


def generate_exploration_plot(dataset_split, class_counts, sizes):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 18))
    fig.suptitle("Etapa 1 — Exploración del Dataset ADE20K (scene_parse_150)", fontsize=16, fontweight='bold')

    items = list(dataset_split)[:8]
    for i, item in enumerate(items):
        img = np.array(item["image"].convert("RGB"))
        mask = np.array(item["annotation"])

        ax_img = fig.add_subplot(4, 6, 2 * i + 1 + (i // 4) * 2 if i < 4 else 2 * (i - 4) + 13)
        ax_img = fig.add_subplot(6, 4, i + 1)
        ax_img.imshow(img)
        ax_img.set_title(f"Imagen {i+1}", fontsize=8)
        ax_img.axis("off")

        ax_mask = fig.add_subplot(6, 4, i + 5)
        ax_mask.imshow(mask, cmap="tab20", vmin=0, vmax=149)
        ax_mask.set_title(f"Máscara {i+1}", fontsize=8)
        ax_mask.axis("off")

    top_idx = np.argsort(class_counts)[-30:][::-1]
    ax_hist = fig.add_subplot(3, 1, 3)
    colors = plt.cm.tab20(np.linspace(0, 1, 30))
    class_names = [ADE20K_CLASSES[i] if i < len(ADE20K_CLASSES) else f"cls_{i}" for i in top_idx]
    ax_hist.bar(range(30), class_counts[top_idx], color=colors)
    ax_hist.set_xticks(range(30))
    ax_hist.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax_hist.set_title("Top 30 clases por número de píxeles", fontsize=11)
    ax_hist.set_ylabel("Píxeles totales")

    plt.tight_layout()
    out_path = PLOTS_DIR / "etapa1_exploracion.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\nFigura guardada en {out_path}")


def print_summary(counts_per_split: dict, class_counts: np.ndarray, sizes: list):
    print("\n" + "=" * 60)
    print("RESUMEN ETAPA 1")
    print("=" * 60)
    for split, count in counts_per_split.items():
        print(f"  {split:10s}: {count} imágenes")

    present_classes = np.where(class_counts > 0)[0]
    print(f"\n  Clases presentes : {len(present_classes)} / 150")
    print(f"  Clases totales   : 150")

    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"  Resolución media : {np.mean(widths):.0f} x {np.mean(heights):.0f} px")
        print(f"  Resolución min   : {min(widths)} x {min(heights)} px")
        print(f"  Resolución max   : {max(widths)} x {max(heights)} px")
    print("=" * 60)


def main():
    print("=" * 60)
    print("ETAPA 1: Descarga y exploración del dataset ADE20K")
    print("=" * 60)

    print("\nCargando dataset desde HuggingFace...")
    ds = load_dataset("scene_parse_150", trust_remote_code=True)

    n_train = save_split(ds["train"], "train", TRAIN_SIZE)
    n_val = save_split(ds["validation"], "validation", VAL_SIZE)

    # scene_parse_150 has no dedicated test split; use last part of validation
    val_items = list(ds["validation"])
    test_items = val_items[:TEST_SIZE]

    test_dir = DATA_RAW / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGuardando split 'test' ({TEST_SIZE} imágenes)...")
    for i, item in enumerate(tqdm(test_items, desc="test")):
        img_path = test_dir / f"img_{i:06d}.jpg"
        mask_path = test_dir / f"mask_{i:06d}.png"

        img = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(img_path, "JPEG", quality=95)

        mask = item["annotation"]
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.array(mask, dtype=np.uint8))
        mask.save(mask_path, "PNG")
    n_test = TEST_SIZE

    copy_samples("train")

    print("\nCalculando estadísticas del dataset...")
    class_counts, sizes = compute_stats(ds["train"], max_items=500)

    generate_exploration_plot(ds["train"], class_counts, sizes)

    counts_per_split = {"train": n_train, "validation": n_val, "test": n_test}
    print_summary(counts_per_split, class_counts, sizes)

    meta = {
        "splits": counts_per_split,
        "num_classes": 150,
        "class_names": ADE20K_CLASSES,
        "mean_width": float(np.mean([s[0] for s in sizes])) if sizes else 0,
        "mean_height": float(np.mean([s[1] for s in sizes])) if sizes else 0,
    }
    meta_path = DATA_RAW / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadatos guardados en {meta_path}")


if __name__ == "__main__":
    main()
