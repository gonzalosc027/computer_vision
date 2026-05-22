import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.visualization import draw_detections
from utils.traffic_metrics import (
    get_detection_stats,
    calculate_traffic_density,
    classify_congestion,
    CLASS_NAMES,
)

TEST_IMGS = ROOT / "data" / "processed" / "test" / "images"
OUT_DIR = ROOT / "results" / "stage4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT / "models"


def load_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


def predict_image(model, img_path: str | Path, conf: float = 0.35) -> dict:
    """Ejecuta inferencia sobre una imagen. Devuelve detecciones + estadísticas."""
    img_path = Path(img_path)
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar {img_path}")

    h, w = img_bgr.shape[:2]
    t0 = time.perf_counter()
    results = model.predict(str(img_path), conf=conf, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown",
            "confidence": float(box.conf[0]),
            "class_id": cls_id,
            "box": {
                "x1": float(box.xyxy[0][0]),
                "y1": float(box.xyxy[0][1]),
                "x2": float(box.xyxy[0][2]),
                "y2": float(box.xyxy[0][3]),
            },
        })

    stats = get_detection_stats(detections)
    density = calculate_traffic_density(detections, w, h)
    congestion_label, congestion_color = classify_congestion(density)
    annotated_bgr = draw_detections(img_bgr, detections)

    return {
        "image_path": str(img_path),
        "detections": detections,
        "stats": stats,
        "density": density,
        "congestion": congestion_label,
        "inference_ms": round(elapsed_ms, 1),
        "annotated_bgr": annotated_bgr,
        "original_bgr": img_bgr,
    }


def run_batch_inference(model, img_dir: Path, conf: float = 0.35, max_imgs: int = 20) -> list[dict]:
    """Procesamiento por lotes."""
    paths = sorted(img_dir.glob("*.jpg"))[:max_imgs]
    if not paths:
        print(f"  No hay imágenes en {img_dir}")
        return []

    results = []
    print(f"  Procesando {len(paths)} imágenes …")
    for p in paths:
        try:
            r = predict_image(model, p, conf=conf)
            results.append(r)
        except Exception as e:
            print(f"    Error en {p.name}: {e}")
    return results


def save_annotated_images(results: list[dict], out_subdir: str = "annotated") -> None:
    out = OUT_DIR / out_subdir
    out.mkdir(exist_ok=True)
    for r in results:
        stem = Path(r["image_path"]).stem
        cv2.imwrite(str(out / f"{stem}_detected.jpg"), r["annotated_bgr"])
    print(f"  Imágenes anotadas guardadas en {out}")


def plot_comparison_grid(results_base: list[dict], results_ft: list[dict], n: int = 4) -> None:
    """Grid comparativo: Original | Modelo base | Fine-tuned."""
    n = min(n, len(results_base), len(results_ft))
    if n == 0:
        return

    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    fig.patch.set_facecolor("#16213e")

    for row in range(n):
        rb = results_base[row]
        rf = results_ft[row]

        for col, (img_bgr, title) in enumerate([
            (rb["original_bgr"], "Original"),
            (rb["annotated_bgr"], f"Base  ({rb['stats']['vehicle_count']} veh.)"),
            (rf["annotated_bgr"], f"Fine-tuned ({rf['stats']['vehicle_count']} veh.)"),
        ]):
            ax = axes[row][col] if n > 1 else axes[col]
            ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            if row == 0:
                ax.set_title(title, color="white", fontsize=11, fontweight="bold")

    plt.suptitle("Comparativa de detección: Base vs Fine-tuned", color="white",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "comparison_grid.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Grid comparativo guardado: {out}")


def plot_inference_summary(results: list[dict], label: str) -> None:
    """Resumen estadístico de un batch de inferencias."""
    import pandas as pd

    rows = []
    for r in results:
        row = {"imagen": Path(r["image_path"]).name, "densidad": r["density"],
               "tráfico": r["congestion"], "ms": r["inference_ms"]}
        row.update(r["stats"]["counts"])
        rows.append(row)

    df = pd.DataFrame(rows)
    total_by_class = {cls: int(df[cls].sum()) if cls in df.columns else 0 for cls in CLASS_NAMES}

    print(f"\n  === Resumen de inferencia — {label} ===")
    print(f"  Imágenes procesadas: {len(results)}")
    print(f"  Tiempo medio por imagen: {df['ms'].mean():.1f} ms")
    print(f"  Detecciones totales:")
    for cls, cnt in sorted(total_by_class.items(), key=lambda x: -x[1]):
        if cnt > 0:
            print(f"    {cls:>12}: {cnt}")

    summary_json = OUT_DIR / f"summary_{label.replace(' ', '_').lower()}.json"
    with open(summary_json, "w") as f:
        json.dump({"label": label, "results": [
            {k: v for k, v in r.items() if k not in ("annotated_bgr", "original_bgr")}
            for r in results
        ]}, f, indent=2)
    print(f"  JSON guardado: {summary_json}")


def main() -> None:

    ft_path = MODELS_DIR / "traffic_detector" / "weights" / "best.pt"

    if not TEST_IMGS.exists() or len(list(TEST_IMGS.glob("*.jpg"))) == 0:
        print(f"  No hay imágenes de test en {TEST_IMGS}")
        print("  Ejecuta primero las Etapas 1 y 2.")
        return

    print("  Cargando modelo pre-entrenado (base) …")
    model_base = load_model("yolov8n.pt")

    results_base = run_batch_inference(model_base, TEST_IMGS, conf=0.35, max_imgs=20)
    save_annotated_images(results_base, "annotated_base")
    plot_inference_summary(results_base, "modelo base")

    if ft_path.exists():
        print("\n  Cargando modelo fine-tuned …")
        model_ft = load_model(str(ft_path))
        results_ft = run_batch_inference(model_ft, TEST_IMGS, conf=0.35, max_imgs=20)
        save_annotated_images(results_ft, "annotated_finetuned")
        plot_inference_summary(results_ft, "fine-tuned")
        plot_comparison_grid(results_base, results_ft, n=4)
    else:
        print(f"\n  Modelo fine-tuned no encontrado en {ft_path}")
        print("  (Ejecuta stage3_training/train.py para entrenarlo)")
        print("  Usando solo el modelo base para la inferencia.")

    print(f"\n  Resultados guardados en {OUT_DIR}")
    print("Etapa 4 completada.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inferencia sobre imágenes de tráfico")
    parser.add_argument("--img", type=str, default=None, help="Ruta a imagen específica")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    args = parser.parse_args()

    if args.img:
        model = load_model(args.model)
        r = predict_image(model, args.img, conf=args.conf)
        print(f"\nResultado en {args.img}:")
        print(f"  Vehículos: {r['stats']['vehicle_count']}")
        print(f"  Densidad : {r['density']}%  ({r['congestion']})")
        print(f"  Tiempo   : {r['inference_ms']} ms")
        out = OUT_DIR / ("result_" + Path(args.img).name)
        cv2.imwrite(str(out), r["annotated_bgr"])
        print(f"  Guardado : {out}")
    else:
        main()
