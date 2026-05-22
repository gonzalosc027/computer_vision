import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_YAML = ROOT / "stage3_training" / "data.yaml"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "results" / "stage3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]


def evaluate_model(model_path: str, label: str) -> dict | None:
    from ultralytics import YOLO

    if not Path(model_path).exists() and model_path not in ("yolov8n.pt",):
        print(f"  Modelo no encontrado: {model_path}")
        return None

    print(f"\n  Evaluando: {label} ({model_path})")
    model = YOLO(model_path)
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        verbose=False,
        plots=True,
        save_json=True,
        project=str(OUT_DIR),
        name=label,
        exist_ok=True,
    )
    return {
        "label": label,
        "mAP50":   float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
    }


def plot_comparison(results: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["mAP50", "mAP50_95", "precision", "recall"]
    labels = [r["label"] for r in results]
    colors = ["#4A90D9", "#27AE60"]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#16213e")
    ax.set_facecolor("#1a1a2e")

    for i, (res, color) in enumerate(zip(results, colors)):
        values = [res[m] for m in metrics]
        bars = ax.bar(x + i * width - width / 2, values, width, label=res["label"],
                      color=color, alpha=0.9, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", color="white", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color="white", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Valor", color="white")
    ax.set_title("Comparativa: Modelo Pre-entrenado vs Fine-tuned", color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Comparativa guardada: {out}")


def print_table(results: list[dict]) -> None:
    print(f"\n{'Modelo':<25} {'mAP@50':>8} {'mAP@50-95':>10} {'Precisión':>10} {'Recall':>8}")
    print("-" * 65)
    for r in results:
        print(f"{r['label']:<25} {r['mAP50']:>8.3f} {r['mAP50_95']:>10.3f} {r['precision']:>10.3f} {r['recall']:>8.3f}")


def main() -> None:

    if not DATA_YAML.exists():
        print("  ERROR: data.yaml no encontrado. Ejecuta primero la Etapa 2.")
        sys.exit(1)

    finetuned_path = str(MODELS_DIR / "traffic_detector" / "weights" / "best.pt")
    pretrained_path = "yolov8n.pt"

    all_results = []

    r_base = evaluate_model(pretrained_path, "Base (pre-entrenado)")
    if r_base:
        all_results.append(r_base)

    r_ft = evaluate_model(finetuned_path, "Fine-tuned (tráfico)")
    if r_ft:
        all_results.append(r_ft)

    if not all_results:
        print("  No se pudo evaluar ningún modelo.")
        return

    print_table(all_results)

    if len(all_results) == 2:
        delta_map = all_results[1]["mAP50"] - all_results[0]["mAP50"]
        print(f"\n  Mejora del fine-tuning en mAP@50: {delta_map:+.3f}")

    plot_comparison(all_results)

    out_json = OUT_DIR / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Métricas guardadas en {out_json}")
    print("Etapa 3 (evaluación) completada.")


if __name__ == "__main__":
    main()
