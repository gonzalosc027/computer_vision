import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"


def run_step(name: str, script: str, description: str):
    print(f"\n{'='*60}")
    print(f"  {name}: {description}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(SRC_DIR / script)],
        cwd=str(BASE_DIR),
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] {name} falló con código {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[OK] {name} completada en {elapsed:.1f}s")


def run_tests(test_file: str):
    print(f"\nEjecutando tests: {test_file}")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(BASE_DIR / "tests" / test_file), "-v", "--tb=short"],
        cwd=str(BASE_DIR),
    )
    if result.returncode != 0:
        print(f"\n[ADVERTENCIA] Algunos tests fallaron en {test_file}")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Pipeline de segmentación semántica")
    parser.add_argument("--skip-train", action="store_true", help="Saltar la etapa de entrenamiento")
    parser.add_argument("--only-app", action="store_true", help="Solo lanzar la aplicación Gradio")
    parser.add_argument("--skip-download", action="store_true", help="Saltar descarga de datos")
    args = parser.parse_args()

    print("=" * 60)
    print("  PIPELINE DE SEGMENTACIÓN SEMÁNTICA — ADE20K")
    print("=" * 60)
    start_total = time.time()

    if args.only_app:
        run_step("ETAPA 5", "etapa5_app.py", "Aplicación Gradio")
        return

    if not args.skip_download:
        run_step("ETAPA 1", "etapa1_datos.py", "Descarga y exploración del dataset")
        run_tests("test_etapa1.py")

    run_step("ETAPA 2", "etapa2_preprocesado.py", "Preprocesado y transformaciones")
    run_tests("test_etapa2.py")

    if not args.skip_train:
        run_step("ETAPA 3", "etapa3_entrenamiento.py", "Entrenamiento SegFormer-B0")
        run_tests("test_etapa3.py")

    run_step("ETAPA 4", "etapa4_inferencia.py", "Inferencia y evaluación")
    run_tests("test_etapa4.py")

    total_elapsed = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETADO en {total_elapsed/60:.1f} minutos")
    print(f"{'='*60}")
    print("\nLanzando aplicación Gradio...")
    run_step("ETAPA 5", "etapa5_app.py", "Aplicación Gradio (http://localhost:7860)")


if __name__ == "__main__":
    main()
