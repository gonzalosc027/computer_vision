import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_IMAGES = ROOT / "data" / "raw" / "images"
RAW_ANN = ROOT / "data" / "raw" / "annotations"

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"

# COCO category_id → nombre
VEHICLE_CATS = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}

MAX_IMAGES = 2000


def _download_with_progress(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Descargando {dest.name} …")

    def _hook(count, block_size, total_size):
        if total_size > 0:
            pct = min(int(count * block_size * 100 / total_size), 100)
            sys.stdout.write(f"\r  {pct}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()


def download_annotations() -> Path:
    ann_zip = RAW_ANN / "annotations_val2017.zip"
    ann_json = RAW_ANN / "instances_val2017.json"

    if ann_json.exists():
        print(f"  Anotaciones ya descargadas: {ann_json}")
        return ann_json

    RAW_ANN.mkdir(parents=True, exist_ok=True)
    _download_with_progress(ANN_URL, ann_zip)

    print("  Extrayendo anotaciones …")
    with zipfile.ZipFile(ann_zip, "r") as zf:
        zf.extract("annotations/instances_val2017.json", RAW_ANN)

    extracted = RAW_ANN / "annotations" / "instances_val2017.json"
    extracted.rename(ann_json)
    (RAW_ANN / "annotations").rmdir()
    ann_zip.unlink(missing_ok=True)
    print(f"  Guardado en {ann_json}")
    return ann_json


def filter_vehicle_images(ann_json: Path) -> list[dict]:
    """Retorna metadatos de imágenes que contengan al menos un vehículo."""
    print("  Filtrando imágenes con vehículos …")
    with open(ann_json) as f:
        coco = json.load(f)

    cat_ids = set(VEHICLE_CATS.keys())
    relevant_ids: set[int] = set()
    for ann in coco["annotations"]:
        if ann["category_id"] in cat_ids:
            relevant_ids.add(ann["image_id"])

    img_meta = {img["id"]: img for img in coco["images"]}
    selected = [img_meta[i] for i in relevant_ids if i in img_meta]

    # Mezcla reproducible
    import random
    random.seed(42)
    random.shuffle(selected)
    selected = selected[:MAX_IMAGES]

    print(f"  Imágenes con vehículos: {len(relevant_ids)} → usando {len(selected)}")
    return selected


def download_images(image_list: list[dict]) -> None:
    RAW_IMAGES.mkdir(parents=True, exist_ok=True)
    already = {p.name for p in RAW_IMAGES.glob("*.jpg")}
    to_download = [img for img in image_list if img["file_name"] not in already]

    if not to_download:
        print(f"  Todas las imágenes ya están descargadas ({len(image_list)})")
        return

    print(f"  Descargando {len(to_download)} imágenes …")
    failed = []
    for img in tqdm(to_download, unit="img"):
        url = IMG_URL.format(img["id"])
        dest = RAW_IMAGES / img["file_name"]
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            failed.append((img["file_name"], str(e)))

    if failed:
        print(f"  ADVERTENCIA: {len(failed)} imágenes fallaron.")
    print(f"  Descarga completada. Imágenes en {RAW_IMAGES}")


def save_metadata(image_list: list[dict], ann_json: Path) -> None:
    """Guarda metadata filtrada y estadísticas de clases."""
    with open(ann_json) as f:
        coco = json.load(f)

    selected_ids = {img["id"] for img in image_list}
    filtered_anns = [a for a in coco["annotations"] if a["image_id"] in selected_ids]

    class_counts: dict[str, int] = {v: 0 for v in VEHICLE_CATS.values()}
    for ann in filtered_anns:
        name = VEHICLE_CATS.get(ann["category_id"])
        if name:
            class_counts[name] += 1

    meta = {
        "num_images": len(image_list),
        "class_counts": class_counts,
        "images": image_list,
        "annotations": filtered_anns,
    }

    out_file = RAW_ANN / "dataset_meta.json"
    with open(out_file, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nEstadísticas del dataset:")
    print(f"  Imágenes totales : {len(image_list)}")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (cnt // 20)
        print(f"  {cls:>12} : {cnt:>5}  {bar}")
    print(f"  Metadata guardada en {out_file}")


def main() -> None:
    ann_json = download_annotations()
    image_list = filter_vehicle_images(ann_json)
    download_images(image_list)
    save_metadata(image_list, ann_json)
    print("\nEtapa 1 completada.")


if __name__ == "__main__":
    main()
