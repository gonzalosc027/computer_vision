import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.visualization import (
    draw_detections,
    plot_class_distribution,
    plot_pie_distribution,
    plot_density_gauge,
    plot_video_timeline,
    build_heatmap,
    overlay_heatmap,
    draw_counting_line,
)
from utils.traffic_metrics import (
    get_detection_stats,
    calculate_traffic_density,
    classify_congestion,
    CLASS_NAMES,
    CLASS_COLORS_HEX,
)
from utils.tracking import VehicleTracker, CountingLine
from utils.report import generate_pdf, generate_csv, build_detection_dataframe
from utils.sign_detector import SignDetector, draw_signs, SIGN_TYPES

# COCO class IDs → display name (only traffic-relevant classes)
COCO_TO_NAME: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

MODELS_DIR = ROOT / "models"
FINETUNED_PATH = MODELS_DIR / "traffic_detector" / "weights" / "best.pt"

SAMPLES_IMGS = ROOT / "samples" / "images"
SAMPLES_VIDS = ROOT / "samples" / "videos"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

_sample_images = sorted([p for p in SAMPLES_IMGS.glob("*") if p.suffix.lower() in IMG_EXTS or p.name.endswith(".jpg.webp") or p.name.endswith(".jpeg.webp")]) if SAMPLES_IMGS.exists() else []
_sample_videos = sorted([p for p in SAMPLES_VIDS.glob("*") if p.suffix.lower() in VID_EXTS]) if SAMPLES_VIDS.exists() else []


st.set_page_config(
    page_title="Smart Traffic Analyzer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg,#4A90D9,#27AE60);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom:0;
    }
    .subtitle { color:#666; font-size:1rem; margin-top:0; }
    .speed-green  { color:#1a8c4e; font-weight:700; }
    .speed-yellow { color:#b8860b; font-weight:700; }
    .speed-orange { color:#c85000; font-weight:700; }
    .speed-red    { color:#cc1b1b; font-weight:700; }

    /* Métricas — fondo semitransparente, texto gestionado por el tema de Streamlit */
    [data-testid="metric-container"] {
        background: rgba(74, 144, 217, 0.10) !important;
        border: 1.5px solid rgba(74, 144, 217, 0.35) !important;
        border-radius: 12px;
        padding: 14px 18px;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }

    hr { border-color:#dde3ee; }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_model(model_name: str):
    from ultralytics import YOLO
    if model_name == "Fine-tuned (tráfico)" and FINETUNED_PATH.exists():
        return YOLO(str(FINETUNED_PATH))
    return YOLO("yolov8n.pt")


with st.sidebar:
    st.markdown("## ⚙️ Configuración")

    ft_available = FINETUNED_PATH.exists()
    model_options = ["Pre-entrenado (COCO)"]
    if ft_available:
        model_options.append("Fine-tuned (tráfico)")
    else:
        st.info("Fine-tuned no encontrado.\nEjecuta la Etapa 3.")

    selected_model = st.selectbox("Modelo", model_options)
    conf_thresh = st.slider("Umbral de confianza", 0.10, 0.90, 0.35, 0.05)
    iou_thresh  = st.slider("Umbral IoU (NMS)",    0.10, 0.90, 0.45, 0.05)

    st.markdown("---")
    st.markdown("**🔧 Funciones avanzadas**")
    feat_heatmap  = st.checkbox("🌡️ Heatmap de concentración", value=False)
    feat_counting = st.checkbox("📏 Línea de conteo virtual  *(solo con vídeo)*",  value=False)
    feat_speed    = st.checkbox("⚡ Velocidad estimada  *(solo con vídeo)*",        value=False)
    feat_report   = st.checkbox("📄 Exportar informe",          value=True)
    feat_signs    = True

    if feat_counting:
        line_pct = st.slider("Posición de la línea (%)", 10, 90, 50, 5)
    else:
        line_pct = 50

    if feat_speed:
        st.markdown("**Calibración de escala**")
        scale_m_per_px = st.slider("Metros por píxel", 0.01, 0.50, 0.10, 0.01,
                                    help="Aumenta si las velocidades salen bajas; reduce si salen altas")
        st.caption(f"≈ {scale_m_per_px * 100:.0f} cm/píxel")
        st.caption(
            "💡 **Calibración rápida:** si ves 20 km/h y el vehículo va a 80, "
            "multiplica el valor actual × 4. Si ves 160 y va a 80, divídelo entre 2."
        )
    else:
        scale_m_per_px = 0.10

    st.markdown("---")
    st.markdown("**Filtrar clases:**")
    class_filter = {}
    cols_cb = st.columns(2)
    for i, cls in enumerate(CLASS_NAMES):
        color = CLASS_COLORS_HEX.get(cls, "#888")
        class_filter[cls] = cols_cb[i % 2].checkbox(cls.capitalize(), value=True, key=f"cb_{cls}")

    active_classes = [c for c, v in class_filter.items() if v]
    st.caption(f"Clases activas: {len(active_classes)}/{len(CLASS_NAMES)}")

    st.markdown("---")
    st.markdown("**Leyenda de clases**")
    for cls in CLASS_NAMES:
        color = CLASS_COLORS_HEX.get(cls, "#888")
        st.markdown(f"<span style='color:{color}'>■</span> {cls.capitalize()}",
                    unsafe_allow_html=True)

    if feat_speed:
        st.markdown("---")
        st.markdown("**Leyenda de velocidad**")
        for label, color in [("< 30 km/h", "#27AE60"), ("30–60 km/h", "#F3C800"),
                               ("60–90 km/h", "#E67E22"), ("> 90 km/h", "#E74C3C")]:
            st.markdown(f"<span style='color:{color}'>■</span> {label}", unsafe_allow_html=True)


st.markdown('<p class="main-title">🚦 Smart Traffic Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detección de vehículos · Densidad de tráfico · Velocidad · YOLOv8</p>',
            unsafe_allow_html=True)
st.markdown("---")

model = load_model(selected_model)

@st.cache_resource
def load_sign_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

@st.cache_resource
def get_sign_detector(_version: int = 4):
    return SignDetector()



def _box_iou(a: list, b: list) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(union, 1)


def _predict_frame(img_bgr: np.ndarray, conf: float, use_tracking: bool = False,
                   frame_idx: int = 0) -> tuple[list, float]:
    """Ejecuta predict o track sobre un frame. Devuelve (cajas_raw, elapsed_ms)."""
    h, w = img_bgr.shape[:2]
    img_area = h * w

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img_bgr)
        tmp_path = tmp.name

    t0 = time.perf_counter()
    if use_tracking:
        results = model.track(tmp_path, conf=conf, iou=iou_thresh,
                               persist=True, tracker="bytetrack.yaml", verbose=False)
        main_boxes = list(results[0].boxes)
        os.unlink(tmp_path)
        return main_boxes, (time.perf_counter() - t0) * 1000

    results = model.predict(tmp_path, conf=conf, iou=iou_thresh,
                            classes=None, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    main_boxes = list(results[0].boxes)

    # Segundo pase a umbral bajo SOLO para truck(7) y bus(5),
    # con filtro de tamaño para evitar falsos positivos pequeños.
    truck_conf = max(0.18, conf * 0.50)
    truck_res = model.predict(tmp_path, conf=truck_conf, iou=iou_thresh,
                               classes=[5, 7], verbose=False)
    os.unlink(tmp_path)

    existing = [b.xyxy[0].tolist() for b in main_boxes]
    for eb in truck_res[0].boxes:
        coords = eb.xyxy[0].tolist()
        x1, y1, x2, y2 = coords
        # Solo objetos grandes (≥ 2.5 % de la imagen) → camiones reales, no coches lejanos
        if (x2 - x1) * (y2 - y1) < img_area * 0.025:
            continue
        # No duplicar detecciones existentes
        if any(_box_iou(coords, mc) > 0.30 for mc in existing):
            continue
        main_boxes.append(eb)
        existing.append(coords)

    return main_boxes, elapsed_ms


def _boxes_to_detections(boxes, tracker: VehicleTracker | None, counting_line: CountingLine | None,
                          frame_idx: int, frame_h: int = 0) -> list[dict]:
    from utils.tracking import VEHICLE_REAL_HEIGHT_M, MIN_BOX_HEIGHT_FRAC
    detections = []
    for box in boxes:
        cls_id = int(box.cls[0])
        name = COCO_TO_NAME.get(cls_id)
        if name is None or name not in active_classes:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        track_id = int(box.id[0]) if box.id is not None else None
        speed = None

        if tracker is not None and track_id is not None:
            box_h_px = y2 - y1
            # Sólo calcular velocidad si el vehículo es lo suficientemente grande en imagen
            too_small = frame_h > 0 and (box_h_px / frame_h) < MIN_BOX_HEIGHT_FRAC
            if not too_small:
                # Escala perspectiva: m/px estimados por el tamaño aparente del vehículo
                real_h = VEHICLE_REAL_HEIGHT_M.get(name, 1.5)
                persp_scale = real_h / max(box_h_px, 1)
                speed = tracker.update(track_id, cx, cy, frame_idx, scale_override=persp_scale)

        if counting_line is not None and track_id is not None:
            counting_line.check(track_id, cy)

        detections.append({
            "name": name,
            "confidence": float(box.conf[0]),
            "class_id": cls_id,
            "track_id": track_id,
            "speed": speed,
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })
    return detections


def run_inference_image(img_bgr: np.ndarray) -> dict:
    """Inferencia estática (sin tracking ni velocidad)."""
    h, w = img_bgr.shape[:2]
    boxes, elapsed_ms = _predict_frame(img_bgr, conf_thresh, use_tracking=False)
    detections = _boxes_to_detections(boxes, tracker=None, counting_line=None, frame_idx=0)

    # Reclasificar "car" grandes como "truck": camiones de atrás que YOLO malclasifica
    if "truck" in active_classes:
        img_area = h * w
        for det in detections:
            if det["name"] != "car":
                continue
            bx = det["box"]
            bw = bx["x2"] - bx["x1"]
            bh = bx["y2"] - bx["y1"]
            aspect = bw / max(bh, 1)
            # Grande + forma rectangular cuadrada → camión de atrás
            if bw * bh > img_area * 0.04 and 0.55 < aspect < 1.80:
                det["name"] = "truck"

    stats    = get_detection_stats(detections)
    density  = calculate_traffic_density(detections, w, h)
    cong     = classify_congestion(density)
    hmap     = build_heatmap(detections, img_bgr.shape) if feat_heatmap else None
    heat_img = overlay_heatmap(img_bgr, hmap) if hmap is not None else None
    annotated = draw_detections(img_bgr, detections)

    sd = get_sign_detector(_version=4)

    # Señales regulatorias: SIEMPRE activas (stop · semáforo · límite de velocidad)
    reg_signs = []
    _reg_err = None
    try:
        reg_signs = sd.detect_regulatory_signs(model, img_bgr, conf_thresh)
    except Exception as e:
        _reg_err = str(e)
        reg_signs = []

    def _sign_inside_vehicle(sb: list, vbs: list[list]) -> bool:
        sx1, sy1, sx2, sy2 = sb
        scx, scy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
        s_area = max((sx2 - sx1) * (sy2 - sy1), 1)
        for vb in vbs:
            vx1, vy1, vx2, vy2 = vb
            if vx1 <= scx <= vx2 and vy1 <= scy <= vy2:
                return True
            ix1, iy1 = max(sx1, vx1), max(sy1, vy1)
            ix2, iy2 = min(sx2, vx2), min(sy2, vy2)
            if ix2 > ix1 and iy2 > iy1:
                if (ix2 - ix1) * (iy2 - iy1) / s_area >= 0.40:
                    return True
        return False

    vehicle_boxes_all = [
        [d["box"]["x1"], d["box"]["y1"], d["box"]["x2"], d["box"]["y2"]]
        for d in detections
        if d["name"] in {"car", "truck", "bus", "motorcycle", "bicycle"}
    ]

    # Suprimir señales cuyo centro o área esté dentro de un vehículo detectado
    if reg_signs and vehicle_boxes_all:
        reg_signs = [
            s for s in reg_signs
            if not _sign_inside_vehicle(
                [s["box"]["x1"], s["box"]["y1"], s["box"]["x2"], s["box"]["y2"]],
                vehicle_boxes_all,
            )
        ]

    # Carteles de dirección (azul/verde): solo si el toggle está activo
    dir_signs = []
    if feat_signs:
        try:
            dir_signs = sd.detect_direction_signs(img_bgr)
        except Exception:
            dir_signs = []
        if dir_signs and vehicle_boxes_all:
            dir_signs = [
                s for s in dir_signs
                if not _sign_inside_vehicle(
                    [s["box"]["x1"], s["box"]["y1"], s["box"]["x2"], s["box"]["y2"]],
                    vehicle_boxes_all,
                )
            ]

    signs = reg_signs + dir_signs
    sign_img = draw_signs(annotated, signs) if signs else None

    return {"detections": detections, "stats": stats, "density": density,
            "congestion": cong, "inference_ms": elapsed_ms,
            "annotated": sign_img if sign_img is not None else annotated,
            "annotated_no_signs": annotated,
            "heatmap_img": heat_img, "heatmap": hmap,
            "reg_signs": reg_signs,
            "signs": signs,
            "_reg_err": _reg_err,
            "img_shape": (h, w)}


def _speed_html(speed: float | None) -> str:
    if speed is None:
        return "<span style='color:#888'>—</span>"
    if speed < 30:
        cls = "speed-green"
    elif speed < 60:
        cls = "speed-yellow"
    elif speed < 90:
        cls = "speed-orange"
    else:
        cls = "speed-red"
    return f"<span class='{cls}'>{speed:.0f} km/h</span>"


def show_road_info_panel(reg_signs: list[dict]) -> None:
    """Banner siempre visible con condiciones regulatorias de la vía."""
    if not reg_signs:
        return

    # Eliminar lecturas parciales: si hay "10" y "100", quitar "10" (prefijo de "100")
    speed_vals = [s.get("speed_value") for s in reg_signs if s["type"] == "speed_limit"]
    valid_speeds = {
        v for v in speed_vals
        if v and not any(
            str(other) != str(v) and str(other).startswith(str(v))
            for other in speed_vals
        )
    }

    items = []
    seen_speed: set[int] = set()
    seen_stop = False
    seen_light = False

    for s in reg_signs:
        if s["type"] == "speed_limit":
            v = s.get("speed_value", 0)
            if v and v in valid_speeds and v not in seen_speed:
                seen_speed.add(v)
                items.append({
                    "bg": "#fff0f0", "border": "#e74c3c", "color": "#c0392b",
                    "text": f"🚫 En esta vía el límite de velocidad es de <b>{v} km/h</b>",
                })
        elif s["type"] == "traffic_light" and not seen_light:
            seen_light = True
            items.append({
                "bg": "#fffbf0", "border": "#f39c12", "color": "#9a6b00",
                "text": "🚦 <b>Semáforo</b> presente — respeta la señalización lumínica",
            })

    if not items:
        return

    st.markdown("---")
    st.markdown("### 🛣️ Condiciones de la vía")
    cards_html = "".join(
        f"<div style='background:{it['bg']};border:1.5px solid {it['border']};"
        f"border-radius:10px;padding:10px 18px;margin-bottom:8px;"
        f"font-size:1rem;color:{it['color']};font-weight:600'>{it['text']}</div>"
        for it in items
    )
    st.markdown(cards_html, unsafe_allow_html=True)


def show_detection_table(detections: list[dict], show_speed: bool = False) -> None:
    if not detections:
        return
    st.markdown("#### Detalle de detecciones")
    rows = []
    for i, det in enumerate(detections, 1):
        box = det["box"]
        speed = det.get("speed")
        tid   = det.get("track_id")
        row = {
            "#":          i,
            "Clase":      det["name"].capitalize(),
            "Confianza":  f"{det['confidence']:.1%}",
            "x1": int(box["x1"]), "y1": int(box["y1"]),
            "x2": int(box["x2"]), "y2": int(box["y2"]),
        }
        if show_speed:
            row["Velocidad"] = f"{speed:.0f} km/h" if speed is not None else "—"
            row["ID"] = f"#{tid}" if tid is not None else "—"
        rows.append(row)
    df = pd.DataFrame(rows)
    # Reordenar columnas para poner velocidad visible
    if show_speed:
        cols = ["#", "Clase", "Confianza", "Velocidad", "ID", "x1", "y1", "x2", "y2"]
        df = df[[c for c in cols if c in df.columns]]
    st.dataframe(df, use_container_width=True, hide_index=True)


def _estimate_detection_range(detections: list[dict], img_h: int) -> str | None:
    """
    Estima hasta qué distancia aproximada son fiables las detecciones,
    basándose en el vehículo más pequeño detectado.
    Heurística: un coche típico mide ~1.5 m de alto; con cámara de carretera estándar
    (~50° FoV vertical), a 100 m ocupa ~25-40 px en una imagen 1080p.
    """
    vehicle_names = {"car", "truck", "bus", "motorcycle", "bicycle"}
    heights = [
        (d["box"]["y2"] - d["box"]["y1"]) / img_h
        for d in detections
        if d["name"] in vehicle_names
    ]
    if not heights:
        return None
    min_frac = min(heights)
    if min_frac < 0.03:
        return "~200 m o más"
    if min_frac < 0.06:
        return "~150-200 m"
    if min_frac < 0.10:
        return "~100-150 m"
    if min_frac < 0.15:
        return "~60-100 m"
    if min_frac < 0.25:
        return "~40-60 m"
    return "~20-40 m"


def show_stats_panel(result: dict, show_speed: bool = False) -> None:
    stats = result["stats"]
    density = result["density"]
    cong_label, cong_color = result["congestion"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚗 Vehículos",      stats["vehicle_count"])
    c2.metric("🚶 Peatones",       stats["person_count"])
    c3.metric("📦 Total",          stats["total_count"])
    c4.metric("⏱️ Inferencia",     f"{result['inference_ms']:.0f} ms")

    # Nota sobre el alcance de detección
    img_h = result.get("img_shape", (0, 0))[0]
    det_range = _estimate_detection_range(result["detections"], img_h) if img_h else None
    if det_range and stats["vehicle_count"] > 0:
        st.caption(
            f"ℹ️ Los {stats['vehicle_count']} vehículo{'s' if stats['vehicle_count'] != 1 else ''} "
            f"detectado{'s' if stats['vehicle_count'] != 1 else ''} son los visibles con claridad "
            f"hasta {det_range} aproximadamente. Los vehículos más lejanos aparecen demasiado "
            f"pequeños en imagen y no se detectan con fiabilidad."
        )

    st.markdown("---")

    col_gauge, col_bar, col_pie = st.columns([1.2, 1.5, 1.2])
    with col_gauge:
        st.plotly_chart(plot_density_gauge(density, cong_label, cong_color),
                        use_container_width=True)
    with col_bar:
        st.plotly_chart(plot_class_distribution(stats["counts"]), use_container_width=True)
    with col_pie:
        if stats["total_count"] > 0:
            st.plotly_chart(plot_pie_distribution(stats["counts"]), use_container_width=True)



tab_img, tab_video, tab_cam = st.tabs([
    "🖼️ Imagen", "🎬 Vídeo", "📷 Cámara en vivo"
])

with tab_img:
    st.markdown("### Análisis de imagen")

    img_bgr = None

    if _sample_images:
        sample_labels = ["— Elige un ejemplo —"] + [f"Imagen {i+1}" for i in range(len(_sample_images))]
        chosen = st.selectbox("Ejemplos incluidos", sample_labels, key="sample_img_sel")
        if chosen != "— Elige un ejemplo —":
            idx = sample_labels.index(chosen) - 1
            raw = _sample_images[idx].read_bytes()
            img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    uploaded = st.file_uploader("O sube tu propia foto",
                                type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is not None:

        col_orig, col_det = st.columns(2)
        with col_orig:
            st.markdown("**Original**")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        with st.spinner("Analizando …"):
            result = run_inference_image(img_bgr)

        with col_det:
            if feat_heatmap and result["heatmap_img"] is not None:
                sub_a, sub_b = st.tabs(["Detecciones", "Heatmap"])
                with sub_a:
                    st.image(cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB),
                             use_container_width=True)
                with sub_b:
                    st.image(cv2.cvtColor(result["heatmap_img"], cv2.COLOR_BGR2RGB),
                             use_container_width=True,
                             caption="Mapa de calor — concentración de vehículos")
            else:
                st.markdown(f"**Detectado — {result['stats']['total_count']} objetos**")
                st.image(cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB),
                         use_container_width=True)

        st.markdown("---")
        if feat_speed:
            st.info(
                "⚡ **Velocidad estimada** — la estimación de velocidad requiere **vídeo** "
                "(necesita varios fotogramas consecutivos para calcular el movimiento). "
                "Súbelo en la pestaña **🎬 Vídeo** con la opción activada.",
                icon=None,
            )
        show_stats_panel(result, show_speed=False)

        show_road_info_panel(result.get("reg_signs", []))
        if result.get("_reg_err"):
            st.caption(f"⚠️ Error interno en detección de señales: {result['_reg_err']}")

        dir_signs_only = [s for s in result.get("signs", [])
                          if s["type"] in ("blue", "green")]
        if feat_signs:
            st.markdown("---")
            st.markdown("### 🪧 Carteles de dirección detectados")
            if not dir_signs_only:
                st.info("No se detectaron carteles azules/verdes en esta imagen. "
                        "Prueba con una foto que tenga señales de autopista o carretera.")
            else:
                for i, sign in enumerate(dir_signs_only, 1):
                    conf_str = f"{sign['ocr_conf']:.0%}" if sign.get("ocr_conf", 0) > 0 else "—"
                    reliable_icon = "✅" if sign.get("reliable") else "⚠️ baja confianza"

                    with st.expander(
                        f"{sign['emoji']} Señal #{i} — {sign['label']}  "
                        f"{'  |  Texto: ' + sign['ocr_text'][:40] if sign.get('ocr_text') else ''}",
                        expanded=True,
                    ):
                        col_a, col_b = st.columns([1, 2])
                        with col_a:
                            # Recorte de la señal
                            box = sign["box"]
                            x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
                            crop = img_bgr[y1:y2, x1:x2]
                            if crop.size > 0:
                                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                         use_container_width=True, caption="Recorte de la señal")
                        with col_b:
                            st.markdown(f"**Tipo:** {sign['emoji']} {sign['label']}")
                            color = sign.get("color_hex", "#888")
                            ocr_text  = (sign.get("ocr_text")     or "").strip()
                            raw_text  = (sign.get("raw_ocr_text") or "").strip()
                            show_text = ocr_text or raw_text

                            if show_text:
                                lineas = show_text.split("\n")
                                texto_html = "<br>".join(
                                    f"<span style='font-size:1.05rem;color:{color};"
                                    f"font-weight:700'>{l}</span>"
                                    for l in lineas if l.strip()
                                )
                                if ocr_text:
                                    badge = "✅ Alta confianza" if sign.get("reliable") else "⚠️ Confianza media"
                                else:
                                    badge = "⚠️ Baja confianza — puede contener errores"
                                st.markdown(
                                    f"**Texto leído** — {badge}  \n{texto_html}",
                                    unsafe_allow_html=True,
                                )
                                st.caption(f"Confianza OCR: {conf_str}")
                            else:
                                st.markdown(
                                    "📍 **Señal demasiado lejana o borrosa** — "
                                    "no se ha podido extraer texto con suficiente claridad.",
                                )
                            st.markdown(f"**Significado:**  \n{sign.get('meaning', '—')}")

        with st.expander("🔬 Ver análisis interno del pipeline"):
            st.markdown(
                "Imágenes intermedias generadas durante el procesamiento. "
                "Muestran los filtros, máscaras y transformaciones que usa el sistema "
                "antes de llegar al resultado final."
            )

            # Detecciones brutas YOLO (antes de filtros de clase/tamaño)
            st.markdown("#### Detecciones YOLO")
            col_raw1, col_raw2 = st.columns(2)
            with col_raw1:
                st.image(cv2.cvtColor(result["annotated_no_signs"], cv2.COLOR_BGR2RGB),
                         caption="Detecciones de vehículos (bounding boxes)", use_container_width=True)
            with col_raw2:
                if result.get("heatmap_img") is not None:
                    st.image(cv2.cvtColor(result["heatmap_img"], cv2.COLOR_BGR2RGB),
                             caption="Heatmap — concentración de objetos", use_container_width=True)
                else:
                    st.info("Activa el Heatmap en el sidebar para ver este mapa.")

            # Pipeline de señales
            st.markdown("#### Pipeline de detección de señales")
            try:
                debug_imgs = get_sign_detector(_version=4).build_debug_images(img_bgr)
                keys = list(debug_imgs.keys())
                for i in range(0, len(keys), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        if i + j < len(keys):
                            k = keys[i + j]
                            col.image(cv2.cvtColor(debug_imgs[k], cv2.COLOR_BGR2RGB),
                                      caption=k, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudieron generar las imágenes de debug: {e}")

            # Tabla de detecciones raw con class_id
            st.markdown("#### Detecciones — detalle técnico")
            if result["detections"]:
                import pandas as pd
                rows = []
                for d in result["detections"]:
                    b = d["box"]
                    bw = b["x2"] - b["x1"]
                    bh = b["y2"] - b["y1"]
                    rows.append({
                        "Clase": d["name"],
                        "COCO ID": d.get("class_id", "—"),
                        "Confianza": f"{d['confidence']:.1%}",
                        "Ancho px": f"{bw:.0f}",
                        "Alto px": f"{bh:.0f}",
                        "% imagen": f"{bw * bh / (img_bgr.shape[0] * img_bgr.shape[1]):.1%}",
                        "Track ID": d.get("track_id") or "—",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if feat_report:
            st.markdown("---")
            st.markdown("#### 📥 Exportar resultados")
            dl1, dl2, dl3, _ = st.columns([1, 1, 1, 2])

            _, buf_jpg = cv2.imencode(".jpg", result["annotated"])
            img_filename = uploaded.name if uploaded else "resultado.jpg"
            dl1.download_button("🖼️ Imagen JPG", data=buf_jpg.tobytes(),
                                file_name=f"trafico_{img_filename}", mime="image/jpeg")

            try:
                pdf_bytes = generate_pdf(
                    result["annotated"], result["detections"], result["stats"],
                    result["density"], result["congestion"],
                    heatmap_bgr=result.get("heatmap_img"),
                    signs=result.get("signs"),
                )
                dl2.download_button("📄 Informe PDF", data=pdf_bytes,
                                    file_name="informe_trafico.pdf", mime="application/pdf")
            except Exception as e:
                dl2.warning(f"PDF no disponible: {e}")

            csv_bytes = generate_csv(result["detections"])
            dl3.download_button("📊 Datos CSV", data=csv_bytes,
                                file_name="detecciones.csv", mime="text/csv")
    else:
        st.info("Elige un ejemplo de la lista o sube tu propia foto.")


with tab_video:
    st.markdown("### Análisis de vídeo")

    vid_path_to_use = None

    if _sample_videos:
        vid_labels = ["— Elige un ejemplo —"] + [f"Vídeo {i+1}" for i in range(len(_sample_videos))]
        chosen_vid = st.selectbox("Ejemplos incluidos", vid_labels, key="sample_vid_sel")
        if chosen_vid != "— Elige un ejemplo —":
            idx_v = vid_labels.index(chosen_vid) - 1
            vid_path_to_use = str(_sample_videos[idx_v])

    vid_file = st.file_uploader("O sube tu propio vídeo",
                                type=["mp4", "avi", "mov", "mkv"], key="vid_uploader")

    if vid_file or vid_path_to_use:
        frame_skip = st.slider("Procesar 1 de cada N fotogramas", 1, 10, 2,
                                help="Más alto = más rápido pero menos preciso")

        if feat_counting:
            st.info(f"Línea de conteo al {line_pct}% del alto del fotograma (cian).")

        if feat_speed:
            st.info(
                "⚡ **Velocidad activada** — aparecerá sobre cada vehículo en los fotogramas. "
                "El tracker necesita al menos **4 fotogramas** del mismo vehículo antes de calcular la velocidad. "
                "Ajusta *Metros por píxel* en el panel lateral para calibrar la escala real.",
            )

        if st.button("▶️ Iniciar análisis", type="primary"):
            if vid_file:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(vid_file.read())
                    tmp_path = tmp.name
                _tmp_created = True
            else:
                tmp_path = vid_path_to_use
                _tmp_created = False

            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            # Inicializar herramientas de análisis
            use_tracking = feat_speed or feat_counting
            tracker  = VehicleTracker(fps=fps_vid, scale_m_per_px=scale_m_per_px) if feat_speed    else None
            cline    = CountingLine(y_fraction=line_pct / 100)                      if feat_counting else None
            hmap_acc = np.zeros((frame_h, frame_w), dtype=np.float32)               if feat_heatmap  else None

            if cline:
                cline.set_frame_height(frame_h)

            # Resetear estado de tracking del modelo entre vídeos
            try:
                if hasattr(model, "predictor") and model.predictor is not None:
                    model.predictor = None
            except Exception:
                pass

            ph_video  = st.empty()
            ph_prog   = st.progress(0)
            ph_stats  = st.empty()
            ph_count  = st.empty()

            frame_stats: list[dict] = []
            frame_idx = 0
            processed = 0
            all_detections_last: list[dict] = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                boxes, elapsed_ms = _predict_frame(frame, conf_thresh,
                                                    use_tracking=use_tracking,
                                                    frame_idx=frame_idx)
                detections = _boxes_to_detections(boxes, tracker, cline, frame_idx, frame_h=frame_h)
                all_detections_last = detections
                processed += 1

                # Dibujar
                annotated = draw_detections(frame, detections)
                if feat_heatmap and hmap_acc is not None:
                    hmap_acc = build_heatmap(detections, frame.shape, existing=hmap_acc)
                    annotated = overlay_heatmap(annotated, hmap_acc, alpha=0.30)
                if feat_counting and cline is not None:
                    annotated = draw_counting_line(annotated, cline.y_px,
                                                    cline.count_down, cline.count_up)

                ph_video.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                use_container_width=True,
                                caption=f"Fotograma {frame_idx}/{total_frames} | "
                                        f"{get_detection_stats(detections)['vehicle_count']} veh.")
                ph_prog.progress(min(frame_idx / max(total_frames, 1), 1.0))

                row = {"second": round(frame_idx / fps_vid, 1)}
                row.update(get_detection_stats(detections)["counts"])
                frame_stats.append(row)

                stats_now = get_detection_stats(detections)
                density_now = calculate_traffic_density(detections, frame_w, frame_h)
                cong_now, _ = classify_congestion(density_now)

                with ph_stats.container():
                    cols_s = st.columns(5 if feat_speed else 4)
                    cols_s[0].metric("🚗 Vehículos",  stats_now["vehicle_count"])
                    cols_s[1].metric("📦 Total",      stats_now["total_count"])
                    cols_s[2].metric("📊 Densidad",   f"{density_now:.0f}%")
                    cols_s[3].metric("🚦 Estado",     cong_now)
                    if feat_speed and tracker and detections:
                        speeds = [d["speed"] for d in detections if d.get("speed") is not None]
                        avg_spd = sum(speeds) / len(speeds) if speeds else 0
                        cols_s[4].metric("⚡ Vel. media", f"{avg_spd:.0f} km/h")

                if feat_counting and cline:
                    with ph_count.container():
                        cc1, cc2, cc3 = st.columns(3)
                        cc1.metric("⬇️ Bajando",  cline.count_down)
                        cc2.metric("⬆️ Subiendo", cline.count_up)
                        cc3.metric("📦 Total",    cline.total)

            cap.release()
            if _tmp_created:
                os.unlink(tmp_path)

            st.success(f"Vídeo completado: {processed} fotogramas procesados.")

            # Timeline
            if frame_stats:
                st.plotly_chart(plot_video_timeline(frame_stats), use_container_width=True)

            # Descarga PDF del último frame
            if feat_report and all_detections_last:
                st.markdown("#### 📥 Exportar")
                stats_last = get_detection_stats(all_detections_last)
                density_last = calculate_traffic_density(all_detections_last, frame_w, frame_h)
                pdf_b = generate_pdf(
                    annotated, all_detections_last, stats_last, density_last,
                    classify_congestion(density_last),
                )
                csv_b = generate_csv(all_detections_last)
                dl1, dl2, _ = st.columns([1, 1, 3])
                dl1.download_button("📄 PDF (último frame)", data=pdf_b,
                                    file_name="informe_video.pdf", mime="application/pdf")
                dl2.download_button("📊 CSV detecciones", data=csv_b,
                                    file_name="detecciones_video.csv", mime="text/csv")
    else:
        st.info("Elige un ejemplo de la lista o sube tu propio vídeo.")


with tab_cam:
    st.markdown("### Detección en tiempo real")
    st.warning("Requiere ejecutar la app **localmente** (`streamlit run stage5_application/app.py`).")

    cam_conf = st.slider("Confianza (cámara)", 0.10, 0.90, 0.40, 0.05, key="cam_conf")
    cam_speed = st.checkbox("⚡ Mostrar velocidad en cámara", value=False, key="cam_speed")
    if cam_speed:
        cam_scale = st.slider("Escala (m/px)", 0.01, 0.20, 0.05, 0.01, key="cam_scale")

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    start = col_btn1.button("▶️ Iniciar", type="primary")
    stop  = col_btn2.button("⏹️ Detener")

    if "cam_running" not in st.session_state:
        st.session_state.cam_running = False
    if start:
        st.session_state.cam_running = True
    if stop:
        st.session_state.cam_running = False

    cam_ph = st.empty()
    cam_stats_ph = st.empty()

    if st.session_state.cam_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("No se pudo abrir la cámara.")
            st.session_state.cam_running = False
        else:
            cam_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cam_tracker = VehicleTracker(fps=cam_fps, scale_m_per_px=cam_scale if cam_speed else 0.05)
            cam_hmap = None
            fidx = 0
            while st.session_state.cam_running:
                ret, frame = cap.read()
                if not ret:
                    break
                fidx += 1
                if fidx % 2 != 0:
                    continue

                use_trk = cam_speed
                boxes, _ = _predict_frame(frame, cam_conf, use_tracking=use_trk, frame_idx=fidx)
                h, w = frame.shape[:2]
                dets = _boxes_to_detections(boxes, cam_tracker if cam_speed else None, None, fidx, frame_h=h)
                ann  = draw_detections(frame, dets)

                if feat_heatmap:
                    cam_hmap = build_heatmap(dets, frame.shape, existing=cam_hmap)
                    ann = overlay_heatmap(ann, cam_hmap, alpha=0.25)

                cam_ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                s = get_detection_stats(dets)
                d = calculate_traffic_density(dets, w, h)
                cg, _ = classify_congestion(d)
                with cam_stats_ph.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🚗 Vehículos", s["vehicle_count"])
                    c2.metric("🚶 Personas",  s["person_count"])
                    c3.metric("📊 Densidad",  f"{d:.0f}%")
                    c4.metric("🚦 Estado",    cg)
            cap.release()
            cam_ph.empty()


