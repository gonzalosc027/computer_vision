"""
Pipeline Runner
───────────────
Ejecuta las etapas del proyecto de forma secuencial.

Uso:
    python run_pipeline.py          # etapas 1, 2 y 3
    python run_pipeline.py 1        # solo stage 1
    python run_pipeline.py 2        # solo stage 2
    python run_pipeline.py 3        # solo stage 3
    python run_pipeline.py 4 --image ruta/imagen.jpg
    python run_pipeline.py 5        # lanza la app Streamlit
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def run_stage1():
    from src.stage1_organize import organize_dataset
    organize_dataset()


def run_stage2():
    from src.stage2_preprocess import preprocess_stage
    preprocess_stage()


def run_stage3():
    from src.stage3_train import train
    train()


def run_stage4(image_path: str):
    from src.stage4_inference import run_inference
    run_inference(image_path)


def run_stage5():
    app = ROOT / "src" / "stage5_app.py"
    subprocess.run(["streamlit", "run", str(app)], check=True)


# ── entrypoint ────────────────────────────────────────────────────────────────

args = sys.argv[1:]
stage = args[0] if args else "all"

if stage == "all":
    run_stage1()
    run_stage2()
    run_stage3()
    print("\n  Pipeline stages 1-3 complete.")
    print("  Next steps:")
    print("    Inference : python run_pipeline.py 4 --image tu_imagen.jpg")
    print("    App       : python run_pipeline.py 5\n")

elif stage == "1":
    run_stage1()

elif stage == "2":
    run_stage2()

elif stage == "3":
    run_stage3()

elif stage == "4":
    if "--image" not in args:
        print("ERROR: proporciona la imagen:  python run_pipeline.py 4 --image foto.jpg")
        sys.exit(1)
    img_idx = args.index("--image") + 1
    run_stage4(args[img_idx])

elif stage == "5":
    run_stage5()

else:
    print(__doc__)
