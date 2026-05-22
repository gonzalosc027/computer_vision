import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from utils.traffic_metrics import CLASS_COLORS_BGR, CLASS_COLORS_HEX, CLASS_NAMES


def _speed_bgr(speed: float) -> tuple[int, int, int]:
    if speed < 30:
        return (60, 200, 60)
    if speed < 60:
        return (0, 200, 240)
    if speed < 90:
        return (0, 140, 255)
    return (40, 40, 220)


def draw_detections(img_bgr: np.ndarray, detections: list[dict], min_conf: float = 0.0) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        if det.get("confidence", 1.0) < min_conf:
            continue
        name  = det.get("name", "object")
        conf  = det.get("confidence", 1.0)
        speed = det.get("speed")        # km/h o None
        tid   = det.get("track_id")     # int o None
        box   = det.get("box", {})
        x1, y1 = int(box.get("x1", 0)), int(box.get("y1", 0))
        x2, y2 = int(box.get("x2", w)), int(box.get("y2", h))

        color = CLASS_COLORS_BGR.get(name, (200, 200, 200))
        thickness = 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        id_str = f" #{tid}" if tid is not None else ""
        label = f"{name} {conf:.0%}{id_str}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.52, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3), font, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

        if speed is not None:
            spd_color = _speed_bgr(speed)
            spd_label = f"{speed:.0f} km/h"
            (sw, sh), _ = cv2.getTextSize(spd_label, font, 0.50, 1)
            sy_top = y1 - th - sh - 10
            cv2.rectangle(out, (x1, sy_top - 2), (x1 + sw + 4, sy_top + sh + 2), spd_color, -1)
            cv2.putText(out, spd_label, (x1 + 2, sy_top + sh - 1), font, 0.50, (0, 0, 0), 1, cv2.LINE_AA)

    return out


def plot_class_distribution(counts: dict[str, int]) -> go.Figure:
    classes = [c for c in CLASS_NAMES if counts.get(c, 0) > 0]
    values = [counts[c] for c in classes]
    colors = [CLASS_COLORS_HEX.get(c, "#888") for c in classes]

    fig = go.Figure(go.Bar(
        x=classes,
        y=values,
        marker_color=colors,
        text=values,
        textposition="outside",
    ))
    fig.update_layout(
        title="Distribución de detecciones por clase",
        xaxis_title="Clase",
        yaxis_title="Cantidad",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def plot_pie_distribution(counts: dict[str, int]) -> go.Figure:
    labels = [c for c in CLASS_NAMES if counts.get(c, 0) > 0]
    values = [counts[c] for c in labels]
    colors = [CLASS_COLORS_HEX.get(c, "#888") for c in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.4,
        textinfo="label+percent",
    ))
    fig.update_layout(
        title="Proporción por tipo de objeto",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=20, l=20, r=20),
        showlegend=False,
    )
    return fig


def plot_density_gauge(density: float, label: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=density,
        title={"text": f"Densidad de tráfico<br><span style='color:{color};font-size:14px'>{label}</span>"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 20],  "color": "#2ECC71"},
                {"range": [20, 50], "color": "#F39C12"},
                {"range": [50, 75], "color": "#E67E22"},
                {"range": [75, 100],"color": "#E74C3C"},
            ],
            "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.75, "value": density},
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=60, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_heatmap(detections: list[dict], img_shape: tuple, existing: np.ndarray | None = None) -> np.ndarray:
    h, w = img_shape[:2]
    hmap = existing if existing is not None else np.zeros((h, w), dtype=np.float32)
    for det in detections:
        box = det.get("box", {})
        cx = int((box.get("x1", 0) + box.get("x2", 0)) / 2)
        cy = int((box.get("y1", 0) + box.get("y2", 0)) / 2)
        bw = int(box.get("x2", 0) - box.get("x1", 0))
        bh = int(box.get("y2", 0) - box.get("y1", 0))
        r = max(int(max(bw, bh) / 2.5), 18)
        cx = max(r, min(w - r - 1, cx))
        cy = max(r, min(h - r - 1, cy))
        cv2.circle(hmap, (cx, cy), r, 1.0, -1)
    if hmap.max() > 0:
        hmap = cv2.GaussianBlur(hmap, (0, 0), sigmaX=15)
    return hmap


def overlay_heatmap(img_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if heatmap.max() == 0:
        return img_bgr
    norm = (heatmap / heatmap.max() * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, colored, alpha, 0)


def draw_counting_line(img_bgr: np.ndarray, y_px: int, count_down: int, count_up: int) -> np.ndarray:
    out = img_bgr.copy()
    w = out.shape[1]
    cv2.line(out, (0, y_px), (w, y_px), (0, 255, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, f"v {count_down}", (10, y_px - 8),  font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"^ {count_up}",  (10, y_px + 22), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def plot_video_timeline(frame_stats: list[dict]) -> go.Figure:
    if not frame_stats:
        return go.Figure()

    df = pd.DataFrame(frame_stats)
    fig = go.Figure()
    for cls in CLASS_NAMES:
        if cls in df.columns and df[cls].sum() > 0:
            fig.add_trace(go.Scatter(
                x=df["second"],
                y=df[cls],
                name=cls,
                line=dict(color=CLASS_COLORS_HEX.get(cls, "#888")),
                mode="lines",
            ))
    fig.update_layout(
        title="Vehículos detectados a lo largo del tiempo",
        xaxis_title="Tiempo (s)",
        yaxis_title="Cantidad",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig
