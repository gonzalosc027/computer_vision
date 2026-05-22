import time
import json
import numpy as np
from pathlib import Path
from PIL import Image

import gradio as gr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from etapa1_datos import ADE20K_CLASSES
from etapa4_inferencia import (
    load_model, segmentar_imagen, COLOR_MAP, get_device, MODELS_DIR
)

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "data" / "samples"
DATA_RAW = BASE_DIR / "data" / "raw"
CHECKPOINT_PATH = MODELS_DIR / "best_model.pth"

NUM_CLASSES = 150
OPTION_NVIDIA    = "Pre-entrenado NVIDIA (recomendado)"
OPTION_FINETUNED = "Fine-tuned local"

_device = get_device()
_models = {}


def _load(option: str):
    key = "finetuned" if option == OPTION_FINETUNED else "nvidia"
    if key not in _models:
        ckpt = CHECKPOINT_PATH if key == "finetuned" else None
        _models[key] = load_model(_device, ckpt)
    return _models[key]


def _finetuned_available() -> bool:
    return CHECKPOINT_PATH.exists()


def _get_metrics_description():
    hist_path = MODELS_DIR / "training_history.json"
    if hist_path.exists():
        with open(hist_path) as f:
            hist = json.load(f)
        best_miou = max(hist.get("val_miou", [0]))
        epochs = len(hist.get("train_loss", []))
        return f"{epochs} épocas · val mIoU: <b>{best_miou:.4f}</b>"
    return None


def _color_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _text_color_for_bg(rgb):
    r, g, b = rgb
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#111" if lum > 140 else "#fff"


def generate_legend_html(pred_mask):
    unique, counts = np.unique(pred_mask, return_counts=True)
    total_px = pred_mask.size
    sorted_pairs = sorted(zip(unique, counts), key=lambda x: -x[1])

    chips = []
    for cls_id, cnt in sorted_pairs:
        cls_name = ADE20K_CLASSES[cls_id] if cls_id < len(ADE20K_CLASSES) else f"clase_{cls_id}"
        bg = _color_to_hex(COLOR_MAP[cls_id])
        fg = _text_color_for_bg(COLOR_MAP[cls_id])
        pct = cnt / total_px * 100
        chips.append(
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'background:{bg};color:{fg};padding:4px 10px;border-radius:999px;'
            f'font-size:13px;font-weight:500;margin:3px;white-space:nowrap;">'
            f'{cls_name} <span style="opacity:0.75;font-size:11px;">{pct:.1f}%</span></span>'
        )

    return '<div style="line-height:2.2;">' + "".join(chips) + "</div>"


def generate_stats_html(pred_mask, elapsed, model_label):
    unique, counts = np.unique(pred_mask, return_counts=True)
    total_px = pred_mask.size
    dominant_idx = unique[counts.argmax()]
    dominant_name = ADE20K_CLASSES[dominant_idx] if dominant_idx < len(ADE20K_CLASSES) else f"cls_{dominant_idx}"
    dominant_pct = counts.max() / total_px * 100
    dom_color = _color_to_hex(COLOR_MAP[dominant_idx])

    def card(icon, label, value, sub=""):
        return (
            f'<div style="flex:1;min-width:130px;background:#f8fafc;border:1px solid #e2e8f0;'
            f'border-radius:12px;padding:14px 16px;text-align:center;">'
            f'<div style="font-size:22px;">{icon}</div>'
            f'<div style="font-size:20px;font-weight:700;color:#1e293b;margin:4px 0;">{value}</div>'
            f'<div style="font-size:11px;color:#64748b;font-weight:500;">{label}</div>'
            f'{"<div style=\"font-size:11px;color:#94a3b8;\">" + sub + "</div>" if sub else ""}'
            f'</div>'
        )

    cards = [
        card("🎨", "Clases detectadas", str(len(unique))),
        card("🏆", "Clase dominante",
             f'<span style="color:{dom_color};">{dominant_name}</span>',
             f"{dominant_pct:.1f}% del área"),
        card("⚡", "Inferencia", f"{elapsed*1000:.0f} ms"),
        card("🤖", "Modelo", "NVIDIA" if "NVIDIA" in model_label else "Fine-tuned"),
    ]

    return (
        '<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:4px;">'
        + "".join(cards) + "</div>"
    )


def predict(image_input, alpha_value, model_option):
    empty = '<p style="color:#94a3b8;text-align:center;padding:20px;">Sube una imagen para ver los resultados.</p>'
    if image_input is None:
        return None, None, empty, empty

    if model_option == OPTION_FINETUNED and not _finetuned_available():
        return None, None, '<p style="color:#ef4444;padding:8px;">⚠️ No hay checkpoint local. Ejecuta primero etapa3_entrenamiento.py.</p>', empty

    t0 = time.time()
    model = _load(model_option)

    tmp_path = Path("/tmp/gradio_seg_input.jpg")
    if isinstance(image_input, np.ndarray):
        Image.fromarray(image_input).save(tmp_path)
    else:
        Image.open(image_input).convert("RGB").save(tmp_path)

    original, colored_mask, _, pred_mask = segmentar_imagen(tmp_path, model, _device)
    elapsed = time.time() - t0

    alpha = float(alpha_value) / 100.0
    overlay = (alpha * colored_mask + (1 - alpha) * original).astype(np.uint8)

    return (
        colored_mask,
        overlay,
        generate_stats_html(pred_mask, elapsed, model_option),
        generate_legend_html(pred_mask),
    )


def get_example_images():
    samples = sorted(SAMPLES_DIR.glob("img_*.jpg"))[:6]
    if len(samples) < 6:
        samples += sorted((DATA_RAW / "train").glob("img_*.jpg"))[:6 - len(samples)]
    return [[str(p)] for p in samples[:6]]


CSS = """
body, .gradio-container { font-family: 'Inter', system-ui, sans-serif !important; }

#header {
    background: #1e293b;
    border-radius: 14px;
    padding: 22px 26px 18px;
    margin-bottom: 8px;
}
#header h1 {
    margin: 0 0 4px;
    font-size: 24px;
    font-weight: 700;
    color: #ffffff !important;
}
#header .sub {
    font-size: 13px;
    color: #94a3b8 !important;
    margin: 0 0 10px;
}
#header .badge {
    display: inline-block;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 11px;
    color: #cbd5e1 !important;
    margin-right: 6px;
}
#header .badge-green {
    background: rgba(34,197,94,0.15);
    border-color: rgba(34,197,94,0.3);
    color: #86efac !important;
}

#seg-btn {
    background: #3b82f6 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    color: white !important;
}
#seg-btn:hover { background: #2563eb !important; }

input[type=range] { accent-color: #3b82f6; }

.model-radio label { font-weight: 500; }
"""


def build_app():
    metrics = _get_metrics_description()
    finetuned_ok = _finetuned_available()

    model_choices = [OPTION_NVIDIA]
    if finetuned_ok:
        model_choices.append(OPTION_FINETUNED)

    with gr.Blocks(title="Semantic Segmentation — ADE20K", css=CSS, theme=gr.themes.Base()) as demo:

        ft_badge = ""
        if finetuned_ok and metrics:
            ft_badge = f'<span class="badge badge-green">Fine-tuned · {metrics}</span>'

        gr.HTML(f"""
        <div id="header">
            <h1>🔍 Semantic Segmentation — ADE20K</h1>
            <p class="sub">Segmentación semántica con <strong style="color:#e2e8f0;">150 categorías</strong> de objetos</p>
            <span class="badge">SegFormer-B0</span>
            <span class="badge">ADE20K · 150 clases</span>
            {ft_badge}
        </div>
        """)

        with gr.Row(equal_height=False):

            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Imagen de entrada")
                image_input = gr.Image(
                    label=None, type="numpy", show_label=False,
                    height=300, sources=["upload", "clipboard"],
                )

                gr.Markdown("### Modelo")
                model_selector = gr.Radio(
                    choices=model_choices,
                    value=OPTION_NVIDIA,
                    label=None,
                    show_label=False,
                    elem_classes=["model-radio"],
                    info="El fine-tuned local requiere haber ejecutado la Etapa 3" if not finetuned_ok else "",
                )

                alpha_slider = gr.Slider(
                    minimum=0, maximum=100, value=60, step=5,
                    label="Opacidad overlay (%)",
                )

                submit_btn = gr.Button(
                    "▶  Segmentar imagen", variant="primary",
                    elem_id="seg-btn", size="lg",
                )

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("🎨  Segmentación"):
                        mask_output = gr.Image(
                            label=None, type="numpy", show_label=False, height=370,
                        )
                    with gr.TabItem("🖼  Overlay"):
                        overlay_output = gr.Image(
                            label=None, type="numpy", show_label=False, height=370,
                        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Resumen")
                stats_output = gr.HTML(
                    value='<p style="color:#94a3b8;padding:8px;">Los resultados aparecerán aquí.</p>'
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Leyenda de clases detectadas")
                legend_output = gr.HTML(
                    value='<p style="color:#94a3b8;padding:8px;">Las clases detectadas aparecerán aquí.</p>'
                )

        example_images = get_example_images()
        if example_images:
            gr.Markdown("### Ejemplos")
            gr.Examples(examples=example_images, inputs=image_input,
                        label=None, examples_per_page=6)

        gr.HTML("""
        <div style="text-align:center;padding:14px 0 4px;color:#94a3b8;font-size:11px;border-top:1px solid #e2e8f0;margin-top:8px;">
            SegFormer-B0 · ADE20K scene_parse_150 · 150 clases semánticas
        </div>
        """)

        inputs  = [image_input, alpha_slider, model_selector]
        outputs = [mask_output, overlay_output, stats_output, legend_output]

        submit_btn.click(fn=predict, inputs=inputs, outputs=outputs)
        image_input.change(fn=predict, inputs=inputs, outputs=outputs)
        model_selector.change(fn=predict, inputs=inputs, outputs=outputs)

    return demo


def main():
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)


if __name__ == "__main__":
    main()
