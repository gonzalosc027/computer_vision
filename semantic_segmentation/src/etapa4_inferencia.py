import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from etapa1_datos import ADE20K_CLASSES


def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 150) -> float:
    iou_list = []
    preds = preds.view(-1)
    targets = targets.view(-1)
    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union > 0:
            iou_list.append(intersection / union)
    return float(np.mean(iou_list)) if iou_list else 0.0

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "checkpoints"
DATA_RAW = BASE_DIR / "data" / "raw"
OUTPUTS_TEST = BASE_DIR / "outputs" / "test_results"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"

NUM_CLASSES = 150
IMG_SIZE = 512
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _build_color_map():
    colors = []
    cmaps = [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c, plt.cm.Set1, plt.cm.Set2,
             plt.cm.Set3, plt.cm.Paired, plt.cm.Accent]
    for cmap in cmaps:
        n = cmap.N if hasattr(cmap, 'N') else 20
        for j in range(n):
            r, g, b, _ = cmap(j / max(n - 1, 1))
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
            if len(colors) >= NUM_CLASSES:
                break
        if len(colors) >= NUM_CLASSES:
            break
    while len(colors) < NUM_CLASSES:
        colors.append((128, 128, 128))
    return colors[:NUM_CLASSES]


COLOR_MAP = _build_color_map()
COLOR_ARRAY = np.array(COLOR_MAP, dtype=np.uint8)  # (150, 3)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device, checkpoint_path=None):
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Checkpoint cargado: {checkpoint_path}  (epoch={ckpt.get('epoch', '?')}, mIoU={ckpt.get('val_miou', '?'):.4f})")
    else:
        print("Usando pesos pre-entrenados de HuggingFace (sin fine-tuning local).")
    model.to(device)
    model.eval()
    return model


_TRANSFORM = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    return COLOR_ARRAY[mask]


def segmentar_imagen(ruta_imagen, modelo, device):
    img_pil = Image.open(ruta_imagen).convert("RGB")
    original = np.array(img_pil)
    orig_h, orig_w = original.shape[:2]

    augmented = _TRANSFORM(image=original)
    tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = modelo(pixel_values=tensor)
        logits = outputs.logits  # (1, C, H/4, W/4)
        logits_up = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        pred_mask = logits_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

    colored_mask = mask_to_color(pred_mask)

    alpha = 0.6
    overlay = (alpha * colored_mask + (1 - alpha) * original).astype(np.uint8)

    return original, colored_mask, overlay, pred_mask


def run_test_inference(model, device, save_n=20):
    """Run inference on the test split and compute metrics."""
    test_dir = DATA_RAW / "test"
    img_files = sorted(test_dir.glob("img_*.jpg"))

    OUTPUTS_TEST.mkdir(parents=True, exist_ok=True)

    times = []
    class_intersection = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_union = np.zeros(NUM_CLASSES, dtype=np.int64)

    print(f"\nInferencia sobre {len(img_files)} imágenes de test...")
    for i, img_path in enumerate(tqdm(img_files)):
        mask_path = test_dir / img_path.name.replace("img_", "mask_").replace(".jpg", ".png")
        if not mask_path.exists():
            continue

        t0 = time.time()
        original, colored_mask, overlay, pred_mask = segmentar_imagen(img_path, model, device)
        times.append(time.time() - t0)

        target_mask = np.array(Image.open(mask_path))

        if target_mask.shape != pred_mask.shape:
            target_mask = np.array(Image.fromarray(target_mask).resize(
                (pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST))

        for cls in range(NUM_CLASSES):
            pred_cls = pred_mask == cls
            tgt_cls = target_mask == cls
            class_intersection[cls] += int((pred_cls & tgt_cls).sum())
            class_union[cls] += int((pred_cls | tgt_cls).sum())

        if i < save_n:
            out_img = Image.fromarray(overlay)
            out_img.save(OUTPUTS_TEST / f"overlay_{i:04d}.jpg")

    class_miou = []
    for cls in range(NUM_CLASSES):
        if class_union[cls] > 0:
            class_miou.append(class_intersection[cls] / class_union[cls])
        else:
            class_miou.append(None)

    valid = [v for v in class_miou if v is not None]
    global_miou = float(np.mean(valid)) if valid else 0.0

    fps = len(times) / sum(times) if times else 0

    return global_miou, class_miou, fps


def generate_results_grid(model, device):
    """4-column grid: original | mask | overlay for 4 test images."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    test_dir = DATA_RAW / "test"
    img_files = sorted(test_dir.glob("img_*.jpg"))[:4]

    if not img_files:
        return

    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    fig.suptitle("Etapa 4 — Resultados de Inferencia", fontsize=14, fontweight='bold')
    col_titles = ["Imagen Original", "Segmentación", "Overlay"]

    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=12, fontweight='bold')

    for row, img_path in enumerate(img_files):
        original, colored_mask, overlay, _ = segmentar_imagen(img_path, model, device)
        for col, img_data in enumerate([original, colored_mask, overlay]):
            axes[row][col].imshow(img_data)
            axes[row][col].axis("off")

    plt.tight_layout()
    out_path = PLOTS_DIR / "etapa4_resultados.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Grid de resultados guardado en {out_path}")


def generate_class_metrics_plot(class_miou: list):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    valid = [(i, v) for i, v in enumerate(class_miou) if v is not None]
    valid.sort(key=lambda x: x[1], reverse=True)

    top10 = valid[:10]
    bot10 = valid[-10:]
    shown = top10 + bot10

    indices = [x[0] for x in shown]
    values = [x[1] for x in shown]
    names = [ADE20K_CLASSES[i] if i < len(ADE20K_CLASSES) else f"cls_{i}" for i in indices]
    colors = ["green"] * 10 + ["red"] * 10

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(shown)), values, color=colors)
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("IoU")
    ax.set_title("Etapa 4 — mIoU por clase (Top 10 mejores y peores)", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=np.mean([v for v in values if v is not None]), color="blue", linestyle="--", label="Media")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    out_path = PLOTS_DIR / "etapa4_metricas_por_clase.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Métricas por clase guardadas en {out_path}")


def main():
    print("=" * 60)
    print("ETAPA 4: Inferencia")
    print("=" * 60)

    device = get_device()
    print(f"Dispositivo: {device}")

    checkpoint = MODELS_DIR / "best_model.pth"
    model = load_model(device, checkpoint)

    global_miou, class_miou, fps = run_test_inference(model, device, save_n=20)

    print(f"\nmIoU global en test  : {global_miou:.4f}")
    print(f"Velocidad inferencia : {fps:.2f} imágenes/segundo")

    valid_class = [(i, v) for i, v in enumerate(class_miou) if v is not None]
    valid_class.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 10 mejores clases:")
    for i, v in valid_class[:10]:
        name = ADE20K_CLASSES[i] if i < len(ADE20K_CLASSES) else f"cls_{i}"
        print(f"  {name:30s}: {v:.4f}")
    print("\nPeores 10 clases:")
    for i, v in valid_class[-10:]:
        name = ADE20K_CLASSES[i] if i < len(ADE20K_CLASSES) else f"cls_{i}"
        print(f"  {name:30s}: {v:.4f}")

    generate_results_grid(model, device)
    generate_class_metrics_plot(class_miou)

    print("\nEtapa 4 completada.")


if __name__ == "__main__":
    main()
