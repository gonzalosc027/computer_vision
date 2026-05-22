import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_YAML = ROOT / "stage3_training" / "data.yaml"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def check_prerequisites() -> bool:
    if not DATA_YAML.exists():
        print(f"  ERROR: No se encontró {DATA_YAML}")
        print("  Ejecuta primero: python stage2_preprocessing/convert_annotations.py")
        return False

    train_imgs = ROOT / "data" / "processed" / "train" / "images"
    n = len(list(train_imgs.glob("*.jpg")))
    if n == 0:
        print(f"  ERROR: No hay imágenes en {train_imgs}")
        return False

    print(f"  Imágenes de entrenamiento encontradas: {n}")
    return True


def train(
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 8,
    model_size: str = "n",  # n=nano, s=small, m=medium
    use_augmented: bool = True,
) -> Path:
    from ultralytics import YOLO
    import yaml

    # Si hay datos aumentados, actualiza el yaml para usarlos
    if use_augmented:
        aug_imgs = ROOT / "data" / "augmented" / "images"
        if len(list(aug_imgs.glob("*.jpg"))) > 0:
            # Crea un yaml combinado
            combined_yaml = ROOT / "stage3_training" / "data_augmented.yaml"
            with open(DATA_YAML) as f:
                data_cfg = yaml.safe_load(f)
            data_cfg["train"] = str(ROOT / "data" / "processed" / "train" / "images") + \
                                 f"\n  - {str(aug_imgs)}"
            with open(combined_yaml, "w") as f:
                yaml.dump(data_cfg, f)
            active_yaml = combined_yaml
            print("  Usando dataset aumentado para entrenamiento.")
        else:
            active_yaml = DATA_YAML
            print("  Aumentación no encontrada, usando dataset original.")
    else:
        active_yaml = DATA_YAML

    model = YOLO(f"yolov8{model_size}.pt")

    print(f"  Modelo: YOLOv8{model_size} | Épocas: {epochs} | Batch: {batch} | Imgsz: {imgsz}")

    results = model.train(
        data=str(active_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="traffic_detector",
        project=str(MODELS_DIR),
        save=True,
        patience=10,
        plots=True,
        verbose=True,
        exist_ok=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
    )

    best_model = MODELS_DIR / "traffic_detector" / "weights" / "best.pt"
    print(f"\n  Mejor modelo guardado: {best_model}")
    return best_model


def main() -> None:

    if not check_prerequisites():
        sys.exit(1)

    print("  Iniciando fine-tuning de YOLOv8n …")
    print("  (En CPU ~20-40 min. Con GPU <5 min)\n")

    best = train(epochs=30, imgsz=640, batch=8)
    print(f"\n  Entrenamiento completado. Modelo: {best}")
    print("Etapa 3 (entrenamiento) completada.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entrenar YOLOv8 en dataset de tráfico")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--batch",  type=int, default=8)
    parser.add_argument("--model",  type=str, default="n", choices=["n", "s", "m"])
    args = parser.parse_args()
    main()
