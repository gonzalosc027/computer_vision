import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_IMAGES = ROOT / "data" / "raw" / "images"
META_FILE = ROOT / "data" / "raw" / "annotations" / "dataset_meta.json"
PROC = ROOT / "data" / "processed"

# COCO cat_id → YOLOv8 class index
COCO_TO_YOLO = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 8: 5}
CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

SPLITS = {"train": 0.70, "val": 0.20, "test": 0.10}


def coco_bbox_to_yolo(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, nw)),
        max(0.0, min(1.0, nh)),
    )


def create_split_dirs() -> None:
    for split in SPLITS:
        (PROC / split / "images").mkdir(parents=True, exist_ok=True)
        (PROC / split / "labels").mkdir(parents=True, exist_ok=True)


def build_ann_map(annotations: list[dict]) -> dict[int, list[dict]]:
    ann_map: dict[int, list[dict]] = {}
    for ann in annotations:
        ann_map.setdefault(ann["image_id"], []).append(ann)
    return ann_map


def convert_and_split(meta: dict) -> dict[str, int]:
    images = meta["images"]
    annotations = meta["annotations"]

    # Solo imágenes que existen en disco
    available = [img for img in images if (RAW_IMAGES / img["file_name"]).exists()]
    print(f"  Imágenes disponibles en disco: {len(available)}")

    random.seed(42)
    random.shuffle(available)

    n = len(available)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])

    split_groups = {
        "train": available[:n_train],
        "val":   available[n_train:n_train + n_val],
        "test":  available[n_train + n_val:],
    }

    ann_map = build_ann_map(annotations)
    counts: dict[str, int] = {}

    for split, img_list in split_groups.items():
        counts[split] = len(img_list)
        for img_meta in img_list:
            img_id = img_meta["id"]
            src_img = RAW_IMAGES / img_meta["file_name"]
            stem = Path(img_meta["file_name"]).stem

            dst_img = PROC / split / "images" / img_meta["file_name"]
            dst_lbl = PROC / split / "labels" / f"{stem}.txt"

            shutil.copy2(src_img, dst_img)

            lines = []
            for ann in ann_map.get(img_id, []):
                cls_id = COCO_TO_YOLO.get(ann["category_id"])
                if cls_id is None:
                    continue
                if ann.get("area", 1) < 100:
                    continue
                cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], img_meta["width"], img_meta["height"])
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            dst_lbl.write_text("\n".join(lines))

    return counts


def write_data_yaml() -> Path:
    yaml_path = ROOT / "stage3_training" / "data.yaml"
    content = (
        f"path: {str(PROC)}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    yaml_path.write_text(content)
    print(f"  data.yaml guardado en {yaml_path}")
    return yaml_path


def main() -> None:

    if not META_FILE.exists():
        print(f"  No se encontró {META_FILE}. Ejecuta primero la Etapa 1.")
        sys.exit(1)

    with open(META_FILE) as f:
        meta = json.load(f)

    create_split_dirs()
    counts = convert_and_split(meta)
    yaml_path = write_data_yaml()

    for split, n in counts.items():
        print(f"  {split}: {n} imágenes")

    print(f"\n  Formato: YOLOv8 TXT (class cx cy w h, normalizado)")
    print(f"  Dataset procesado en {PROC}")
    print("Etapa 2 (conversión) completada.")


if __name__ == "__main__":
    main()
