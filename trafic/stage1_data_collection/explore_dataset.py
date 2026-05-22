import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_IMAGES = ROOT / "data" / "raw" / "images"
META_FILE = ROOT / "data" / "raw" / "annotations" / "dataset_meta.json"
OUT_DIR = ROOT / "results" / "stage1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = {
    "person": "#F3C800",
    "bicycle": "#FF55CC",
    "car": "#4A90D9",
    "motorcycle": "#3CB371",
    "bus": "#E67E22",
    "truck": "#E74C3C",
}


def plot_class_distribution(class_counts: dict[str, int]) -> None:
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = [CLASS_COLORS.get(c, "#888") for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(classes, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("Distribución de clases en el dataset descargado", fontsize=14, fontweight="bold")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Número de instancias")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(cnt), ha="center", va="bottom", color="white", fontsize=11)

    plt.tight_layout()
    out = OUT_DIR / "class_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico guardado: {out}")


def plot_image_size_distribution(images: list[dict]) -> None:
    widths = [img["width"] for img in images]
    heights = [img["height"] for img in images]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#16213e")

    for ax, values, title, color in zip(
        axes,
        [widths, heights],
        ["Distribución de anchos (px)", "Distribución de alturas (px)"],
        ["#4A90D9", "#E67E22"],
    ):
        ax.hist(values, bins=20, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_title(title, color="white", fontsize=12)
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "image_sizes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico guardado: {out}")


def plot_sample_grid(images: list[dict], annotations: list[dict]) -> None:
    """Muestra una cuadrícula 3×4 de imágenes con bounding boxes."""
    ann_by_img: dict[int, list] = {}
    for ann in annotations:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    sample_ids = [img["id"] for img in images if (RAW_IMAGES / img["file_name"]).exists()][:12]
    n = len(sample_ids)
    if n == 0:
        print("  No hay imágenes locales para mostrar.")
        return

    cols, rows = 4, 3
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#16213e")

    img_by_id = {img["id"]: img for img in images}

    COCO_TO_NAME = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
    NAME_TO_BGR = {
        "person": (0, 200, 243), "bicycle": (153, 85, 255),
        "car": (217, 144, 74), "motorcycle": (60, 220, 160),
        "bus": (50, 130, 240), "truck": (50, 50, 220),
    }

    for idx, img_id in enumerate(sample_ids):
        ax = fig.add_subplot(rows, cols, idx + 1)
        meta = img_by_id[img_id]
        path = RAW_IMAGES / meta["file_name"]
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        for ann in ann_by_img.get(img_id, []):
            name = COCO_TO_NAME.get(ann["category_id"])
            if not name:
                continue
            x, y, w, h = [int(v) for v in ann["bbox"]]
            color = NAME_TO_BGR.get(name, (200, 200, 200))
            cv2.rectangle(rgb, (x, y), (x + w, y + h), color, 2)

        ax.imshow(rgb)
        ax.axis("off")
        ax.set_title(f"id:{img_id}", color="white", fontsize=7)

    plt.suptitle("Muestra del dataset con anotaciones", color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "sample_grid.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Cuadrícula guardada: {out}")


def print_summary(meta: dict) -> None:
    print("\n  === Resumen del dataset ===")
    print(f"  Total imágenes   : {meta['num_images']}")
    print(f"  Total anotaciones: {len(meta['annotations'])}")
    print("\n  Instancias por clase:")
    for cls, cnt in sorted(meta["class_counts"].items(), key=lambda x: -x[1]):
        pct = cnt / max(sum(meta["class_counts"].values()), 1) * 100
        print(f"    {cls:>12} : {cnt:>5}  ({pct:.1f}%)")


def main() -> None:

    if not META_FILE.exists():
        print(f"  No se encontró {META_FILE}. Ejecuta primero download_coco_subset.py")
        return

    with open(META_FILE) as f:
        meta = json.load(f)

    print_summary(meta)
    plot_class_distribution(meta["class_counts"])
    plot_image_size_distribution(meta["images"])
    plot_sample_grid(meta["images"], meta["annotations"])
    print(f"\n  Resultados guardados en {OUT_DIR}")
    print("Etapa 1 (EDA) completada.")


if __name__ == "__main__":
    main()
