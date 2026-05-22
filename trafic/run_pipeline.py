import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_script(script: Path, label: str) -> bool:
    print(f"\n{label}")
    result = subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
    if result.returncode != 0:
        print(f"ERROR en {script.name} (código {result.returncode})")
        return False
    return True


STAGES = {
    1: [
        (ROOT / "stage1_data_collection" / "download_coco_subset.py",
         "ETAPA 1a: Descarga del dataset COCO"),
        (ROOT / "stage1_data_collection" / "explore_dataset.py",
         "ETAPA 1b: Exploración y estadísticas del dataset"),
    ],
    2: [
        (ROOT / "stage2_preprocessing" / "convert_annotations.py",
         "ETAPA 2a: Conversión COCO → YOLOv8 + split"),
        (ROOT / "stage2_preprocessing" / "augmentation_pipeline.py",
         "ETAPA 2b: Aumentación de datos"),
    ],
    3: [
        (ROOT / "stage3_training" / "train.py",
         "ETAPA 3a: Entrenamiento (fine-tuning YOLOv8n)"),
        (ROOT / "stage3_training" / "evaluate.py",
         "ETAPA 3b: Evaluación comparativa"),
    ],
    4: [
        (ROOT / "stage4_inference" / "inference.py",
         "ETAPA 4: Inferencia sobre imágenes de test"),
    ],
}


def launch_app() -> None:
    print("\nArrancando aplicación …")
    app_path = ROOT / "stage5_application" / "app.py"
    subprocess.run(
        ["streamlit", "run", str(app_path), "--server.port", "8501"],
        cwd=str(ROOT),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Traffic Analyzer — Pipeline completo")
    parser.add_argument("--stage", nargs="+", type=int, choices=[1, 2, 3, 4],
                        help="Etapas a ejecutar (1-4)")
    parser.add_argument("--app", action="store_true", help="Lanzar solo la app Streamlit")
    parser.add_argument("--skip-train", action="store_true",
                        help="Omitir entrenamiento (usa modelo pre-entrenado)")
    args = parser.parse_args()

    if args.app:
        launch_app()
        return

    stages_to_run = args.stage if args.stage else [1, 2, 3, 4]

    if args.skip_train and 3 in stages_to_run:
        stages_to_run.remove(3)
        print("  [--skip-train] Etapa 3 omitida. Se usará el modelo pre-entrenado.")

    print(f"\nEtapas a ejecutar: {stages_to_run}")

    for stage_num in stages_to_run:
        for script, label in STAGES[stage_num]:
            ok = run_script(script, label)
            if not ok:
                print(f"\n  Pipeline interrumpido en {label}.")
                sys.exit(1)

    print("\nPipeline completado.")
    print("Para lanzar la app: streamlit run stage5_application/app.py")


if __name__ == "__main__":
    main()
