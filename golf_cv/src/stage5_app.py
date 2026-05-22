"""
Stage 5 – Interactive Streamlit Application
────────────────────────────────────────────
Run:  streamlit run src/stage5_app.py
"""
import io
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CLASS_COLORS_BGR, CLASSES
from stage4_inference import (
    TREE_COLOR_BGR,
    TREE_IDX,
    _clahe_normalise,
    adaptive_zone_reclassify,
    compute_metrics,
    create_confidence_map,
    create_heatmap,
    create_overlay,
    load_model,
    reclassify_trees,
    refine_shapes,
    sliding_window_inference,
    smooth_mowing_stripes,
)

# ── page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis de Campo de Golf",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⛳ Análisis de Campo de Golf")
st.markdown("Evaluación del estado del campo y la cobertura vegetal a partir de imágenes aéreas.")

WATER_IDX = CLASSES.index("water")
OTHER_IDX = CLASSES.index("other")
HEALTHY_IDX = CLASSES.index("healthy_grass")
STRESSED_IDX = CLASSES.index("stressed_grass")

CLASS_LABELS_ES = {
    "healthy_grass":  "Hierba sana",
    "stressed_grass": "Hierba estresada",
    "bunker":         "Bunker",
    "water":          "Agua",
    "other":          "Otro",
    "trees":          "Árboles",
}


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    alpha = st.slider("Opacidad de la segmentación", 0.1, 0.9, 0.45, 0.05)

    st.markdown("---")
    st.subheader("📏 Escala")
    use_scale = st.checkbox("Activar medición de áreas", value=False)
    scale_ha = None
    if use_scale:
        scale_ha = st.number_input(
            "Área total cubierta por la imagen (hectáreas)",
            min_value=0.1, max_value=500.0, value=5.0, step=0.5,
        )
        st.caption("Introduce el área real del terreno visible en la imagen.")

    st.markdown("---")
    st.markdown("**Leyenda de zonas:**")
    for cls, bgr in CLASS_COLORS_BGR.items():
        if cls == "water":
            continue
        r, g, b = bgr[2], bgr[1], bgr[0]
        label = CLASS_LABELS_ES.get(cls, cls)
        st.markdown(
            f'<span style="background:rgb({r},{g},{b});padding:3px 10px;'
            f'border-radius:5px;color:white;font-size:13px">{label}</span>',
            unsafe_allow_html=True,
        )
        st.write("")

    r, g, b = TREE_COLOR_BGR[2], TREE_COLOR_BGR[1], TREE_COLOR_BGR[0]
    st.markdown(
        f'<span style="background:rgb({r},{g},{b});padding:3px 10px;'
        f'border-radius:5px;color:white;font-size:13px">Árboles</span>',
        unsafe_allow_html=True,
    )
    st.write("")


# ── model cache ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo …")
def _get_model():
    try:
        return load_model()
    except FileNotFoundError as e:
        st.error(str(e))
        return None


# ── helpers ───────────────────────────────────────────────────────────────────

def _decode_upload(uploaded_file) -> np.ndarray | None:
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def _run_analysis(model, image_bgr: np.ndarray, alpha: float):
    """Full inference pipeline. Returns (seg_map, prob_map, metrics, overlay, heatmap, conf_map)."""
    img_norm = _clahe_normalise(image_bgr)
    seg_map, prob_map = sliding_window_inference(model, img_norm)
    seg_map = refine_shapes(seg_map)
    seg_map[seg_map == WATER_IDX] = OTHER_IDX
    seg_map = smooth_mowing_stripes(seg_map, image_bgr)
    seg_map = reclassify_trees(seg_map, image_bgr)
    seg_map = adaptive_zone_reclassify(seg_map, image_bgr)
    metrics = compute_metrics(seg_map, image_bgr)
    overlay = create_overlay(image_bgr, seg_map, alpha)
    heatmap = create_heatmap(prob_map, image_bgr.shape[:2])
    conf_map = create_confidence_map(prob_map, image_bgr.shape[:2])
    return seg_map, prob_map, metrics, overlay, heatmap, conf_map


def _show_kpis(metrics: dict):
    health = metrics["health_score"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌿 Índice de Salud",   f"{health:.0f} / 100", delta=f"{health-70:.0f} vs óptimo")
    c2.metric("⚠️ Nivel de Estrés",  metrics["stress_level"])
    c3.metric("📐 Uniformidad",      f"{metrics['uniformity_score']:.0f} / 100")
    c4.metric("🟢 Hierba Sana",      f"{metrics['zone_percentages']['healthy_grass']:.1f}%")
    c5.metric("🟠 Hierba Estresada", f"{metrics['zone_percentages']['stressed_grass']:.1f}%")


def _show_area_table(metrics: dict, scale_ha: float):
    total_m2 = scale_ha * 10_000
    zone_pct  = metrics["zone_percentages"]
    st.subheader("📐 Áreas por Zona")
    rows = []
    for cls, pct in zone_pct.items():
        if cls == "water":
            continue
        area_m2 = pct / 100 * total_m2
        rows.append({
            "Zona": CLASS_LABELS_ES.get(cls, cls),
            "Cobertura (%)": f"{pct:.1f}",
            "Área (m²)": f"{area_m2:,.0f}",
            "Área (ha)": f"{area_m2/10000:.3f}",
        })
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _show_charts(metrics: dict):
    zone_data  = metrics["zone_percentages"]
    health     = metrics["health_score"]
    bar_labels = [CLASS_LABELS_ES.get(k, k) for k in zone_data if k != "water"]
    bar_values = [v for k, v in zone_data.items() if k != "water"]
    bar_colors = [
        f"rgb({v[2]},{v[1]},{v[0]})"
        for k, v in CLASS_COLORS_BGR.items() if k != "water"
    ]

    ch1, ch2 = st.columns(2)
    with ch1:
        st.subheader("Distribución de Zonas")
        fig_pie = go.Figure(go.Pie(
            labels=bar_labels, values=bar_values,
            marker_colors=bar_colors,
            hole=0.38, textinfo="label+percent",
        ))
        fig_pie.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with ch2:
        st.subheader("Indicador de Salud")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=health,
            delta={"reference": 70, "valueformat": ".0f"},
            number={"suffix": " / 100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#2d7a2d", "thickness": 0.25},
                "steps": [
                    {"range": [0,  40], "color": "#ff4444"},
                    {"range": [40, 70], "color": "#ffaa00"},
                    {"range": [70, 100], "color": "#88cc44"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": 70},
            },
        ))
        fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.subheader("Desglose por Zona")
    fig_bar = go.Figure(go.Bar(
        x=bar_labels, y=bar_values, marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in bar_values], textposition="outside",
    ))
    fig_bar.update_layout(yaxis_title="Porcentaje (%)", height=280,
                          margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_bar, use_container_width=True)


def _generate_pdf(image_bgr, overlay, heatmap, conf_map, metrics, filename) -> bytes:
    health = metrics["health_score"]
    zone_data = metrics["zone_percentages"]
    bar_labels = [CLASS_LABELS_ES.get(k, k) for k in zone_data if k != "water"]
    bar_values = [v for k, v in zone_data.items() if k != "water"]
    hex_colors = [
        f"#{v[2]:02x}{v[1]:02x}{v[0]:02x}"
        for k, v in CLASS_COLORS_BGR.items() if k != "water"
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle(
        f"Informe de Análisis de Campo  —  Salud: {health:.0f}/100  |  {metrics['timestamp'][:10]}",
        fontsize=14, fontweight="bold",
    )

    for ax, img, title in zip(
        axes[0],
        [image_bgr, overlay, heatmap, conf_map],
        ["Original", "Segmentación", "Mapa de Salud", "Confianza del Modelo"],
    ):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    axes[1][0].pie(bar_values, labels=bar_labels, autopct="%1.1f%%",
                   colors=hex_colors, startangle=140)
    axes[1][0].set_title("Distribución de Zonas")

    axes[1][1].axis("off")
    summary = (
        f"Índice de Salud:  {health:.0f} / 100\n"
        f"Nivel de Estrés:  {metrics['stress_level']}\n"
        f"Índice de Estrés: {metrics['stress_index']}%\n"
        f"Uniformidad:      {metrics['uniformity_score']} / 100\n\n"
        "Observaciones:\n" +
        "\n".join(f"• {r}" for r in metrics["recommendations"])
    )
    axes[1][1].text(
        0.05, 0.97, summary, transform=axes[1][1].transAxes,
        fontsize=9, va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4e8", alpha=0.8),
    )
    axes[1][1].set_title("Resumen", fontsize=11)

    # Zone bar chart
    colors_rgba = [
        tuple(c / 255 for c in [v[2], v[1], v[0]])
        for k, v in CLASS_COLORS_BGR.items() if k != "water"
    ]
    axes[1][2].bar(bar_labels, bar_values, color=colors_rgba)
    axes[1][2].set_ylabel("Cobertura (%)")
    axes[1][2].set_title("Cobertura por Zona")
    axes[1][2].tick_params(axis="x", rotation=20)

    axes[1][3].axis("off")

    handles = [
        mpatches.Patch(color=np.array(v[::-1]) / 255, label=CLASS_LABELS_ES.get(k, k))
        for k, v in CLASS_COLORS_BGR.items() if k != "water"
    ]
    axes[1][3].legend(handles=handles, loc="center", fontsize=10,
                      title="Leyenda", title_fontsize=11, framealpha=0.8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="pdf", bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf.getvalue()


def _create_tree_field_map(seg_map: np.ndarray, shape: tuple) -> np.ndarray:
    """Binary map: trees vs. field in clearly distinct colours."""
    h, w = shape[:2]
    canvas = np.full((h, w, 3), (180, 220, 180), dtype=np.uint8)  # field → pale green
    tree_mask = cv2.resize(
        (seg_map == TREE_IDX).astype(np.uint8),
        (w, h), interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    canvas[tree_mask] = (20, 60, 20)  # trees → dark forest green
    return canvas


def _create_simplified_map(seg_map: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Mapa de zonas:
      Verde      → campo (hierba sana + estresada)
      Amarillo   → bunkers
      Gris oscuro → árboles
      Gris claro  → otro (caminos, edificios…)
    """
    h, w = shape[:2]
    seg_r = cv2.resize(seg_map.astype(np.uint8), (w, h),
                       interpolation=cv2.INTER_NEAREST).astype(np.int32)

    canvas = np.full((h, w, 3), 210, dtype=np.uint8)  # fondo gris claro = "otro"

    bunker_idx   = CLASSES.index("bunker")
    healthy_idx  = CLASSES.index("healthy_grass")
    stressed_idx = CLASSES.index("stressed_grass")

    field_mask  = (seg_r == healthy_idx) | (seg_r == stressed_idx)
    bunker_mask = seg_r == bunker_idx
    tree_mask   = seg_r == TREE_IDX

    canvas[field_mask]  = (60, 140, 60)    # verde campo
    canvas[bunker_mask] = (0,  200, 255)   # amarillo bunker
    canvas[tree_mask]   = (40,  70, 40)    # verde oscuro árboles

    return canvas


def _build_diff_map(seg1: np.ndarray, seg2: np.ndarray) -> tuple[np.ndarray, dict]:
    """Compare two seg_maps. Returns colour diff image and change statistics."""
    if seg1.shape != seg2.shape:
        seg2 = cv2.resize(seg2, (seg1.shape[1], seg1.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    diff = np.zeros((*seg1.shape, 3), dtype=np.uint8)
    total = seg1.size

    masks = {
        "deteriorated": (seg1 == HEALTHY_IDX)  & (seg2 == STRESSED_IDX),
        "improved":     (seg1 == STRESSED_IDX) & (seg2 == HEALTHY_IDX),
        "stable_healthy":  (seg1 == HEALTHY_IDX)  & (seg2 == HEALTHY_IDX),
        "stable_stressed": (seg1 == STRESSED_IDX) & (seg2 == STRESSED_IDX),
    }
    colors = {
        "deteriorated":     (0,  0, 220),
        "improved":         (0, 200,  0),
        "stable_healthy":   (34, 139, 34),
        "stable_stressed":  (0, 165, 255),
    }
    stats = {}
    for key, mask in masks.items():
        diff[mask] = colors[key]
        stats[key] = round(float(mask.sum() / total * 100), 2)

    return diff, stats


# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Análisis Individual", "📅 Comparación Temporal"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – Single analysis
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    uploaded = st.file_uploader(
        "📤 Sube una imagen del campo de golf",
        type=["jpg", "jpeg", "png", "webp"],
        key="single",
    )
    if uploaded is None:
        st.info("Sube una imagen para comenzar el análisis.")
        st.stop()

    image_bgr = _decode_upload(uploaded)
    if image_bgr is None:
        st.error("No se pudo decodificar la imagen.")
        st.stop()

    model = _get_model()
    if model is None:
        st.stop()

    with st.spinner("Analizando el campo … (puede tardar 10-30 s)"):
        seg_map, prob_map, metrics, overlay, heatmap, conf_map = _run_analysis(
            model, image_bgr, alpha
        )

    _show_kpis(metrics)
    st.markdown("---")

    # Images row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("Segmentación")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col3:
        st.subheader("Mapa de Salud")
        st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("🔴 Zona dañada → 🟡 Intermedia → 🟢 Zona sana")
    with col4:
        st.subheader("Fiabilidad del análisis")
        st.image(cv2.cvtColor(conf_map, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("🟣 Baja → 🟡 Media → 🟢 Alta")

    st.markdown("---")
    _show_charts(metrics)

    if use_scale and scale_ha:
        st.markdown("---")
        _show_area_table(metrics, scale_ha)

    st.markdown("---")
    st.subheader("📋 Observaciones")
    for rec in metrics["recommendations"]:
        st.info(rec)

    st.markdown("---")
    st.subheader("📄 Descargar Informe PDF")
    pdf_bytes = _generate_pdf(image_bgr, overlay, heatmap, conf_map, metrics, uploaded.name)
    st.download_button(
        label="⬇️ Descargar informe (PDF)",
        data=pdf_bytes,
        file_name=f"golf_report_{Path(uploaded.name).stem}.pdf",
        mime="application/pdf",
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – Temporal comparison
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        "Compara dos imágenes del **mismo campo en fechas distintas** para detectar "
        "zonas que han mejorado o empeorado."
    )
    c1, c2 = st.columns(2)
    with c1:
        up1 = st.file_uploader("📅 Imagen ANTERIOR", type=["jpg", "jpeg", "png", "webp"], key="t1")
    with c2:
        up2 = st.file_uploader("📅 Imagen POSTERIOR", type=["jpg", "jpeg", "png", "webp"], key="t2")

    if up1 is None or up2 is None:
        st.info("Sube las dos imágenes para comparar.")
        st.stop()

    img1 = _decode_upload(up1)
    img2 = _decode_upload(up2)
    if img1 is None or img2 is None:
        st.error("No se pudo decodificar alguna de las imágenes.")
        st.stop()

    model = _get_model()
    if model is None:
        st.stop()

    with st.spinner("Analizando ambas imágenes …"):
        seg1, _, m1, ov1, hm1, cf1 = _run_analysis(model, img1, alpha)
        seg2, _, m2, ov2, hm2, cf2 = _run_analysis(model, img2, alpha)
        diff_map, diff_stats = _build_diff_map(seg1, seg2)

    # Side-by-side KPIs
    st.markdown("### Comparativa de métricas")
    ka1, ka2, ka3 = st.columns(3)
    delta_health = m2["health_score"] - m1["health_score"]
    ka1.metric("Índice de Salud", f"{m2['health_score']:.0f} / 100",
               delta=f"{delta_health:+.0f} vs anterior")
    ka2.metric("Hierba Sana",
               f"{m2['zone_percentages']['healthy_grass']:.1f}%",
               delta=f"{m2['zone_percentages']['healthy_grass'] - m1['zone_percentages']['healthy_grass']:+.1f}%")
    ka3.metric("Hierba Estresada",
               f"{m2['zone_percentages']['stressed_grass']:.1f}%",
               delta=f"{m2['zone_percentages']['stressed_grass'] - m1['zone_percentages']['stressed_grass']:+.1f}%",
               delta_color="inverse")

    st.markdown("---")

    # Images
    st.markdown("### Segmentación")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Anterior")
        st.image(cv2.cvtColor(ov1, cv2.COLOR_BGR2RGB), use_container_width=True)
    with c2:
        st.caption("Posterior")
        st.image(cv2.cvtColor(ov2, cv2.COLOR_BGR2RGB), use_container_width=True)
    with c3:
        st.caption("Mapa de cambios")
        st.image(cv2.cvtColor(diff_map, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Legend for diff map
    legend_html = (
        '<span style="background:#dc0000;padding:2px 8px;border-radius:4px;color:white;font-size:12px">Deterioro</span> '
        '<span style="background:#00c800;padding:2px 8px;border-radius:4px;color:white;font-size:12px">Mejora</span> '
        '<span style="background:#228b22;padding:2px 8px;border-radius:4px;color:white;font-size:12px">Estable sana</span> '
        '<span style="background:#ffa500;padding:2px 8px;border-radius:4px;color:white;font-size:12px">Estable estresada</span>'
    )
    st.markdown(legend_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Estadísticas de cambio")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("🔴 Deterioro",        f"{diff_stats['deteriorated']:.1f}%")
    sc2.metric("🟢 Mejora",           f"{diff_stats['improved']:.1f}%")
    sc3.metric("🌿 Estable sana",     f"{diff_stats['stable_healthy']:.1f}%")
    sc4.metric("🟠 Estable estresada", f"{diff_stats['stable_stressed']:.1f}%")
