import io
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from utils.traffic_metrics import CLASS_COLORS_HEX, CLASS_NAMES


def _speed_color(speed: float | None) -> str:
    if speed is None:
        return "#888888"
    if speed < 30:
        return "#27AE60"
    if speed < 60:
        return "#F39C12"
    if speed < 90:
        return "#E67E22"
    return "#E74C3C"


def build_detection_dataframe(detections: list[dict]) -> pd.DataFrame:
    rows = []
    for i, det in enumerate(detections, 1):
        box = det["box"]
        speed = det.get("speed")
        track_id = det.get("track_id")
        rows.append({
            "#": i,
            "Clase": det["name"].capitalize(),
            "Confianza": f"{det['confidence']:.1%}",
            "Velocidad": f"{speed:.0f} km/h" if speed is not None else "—",
            "ID": f"#{track_id}" if track_id is not None else "—",
            "x1": int(box["x1"]), "y1": int(box["y1"]),
            "x2": int(box["x2"]), "y2": int(box["y2"]),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def generate_pdf(
    annotated_bgr: np.ndarray,
    detections: list[dict],
    stats: dict,
    density: float,
    congestion: tuple[str, str],
    heatmap_bgr: np.ndarray | None = None,
    signs: list[dict] | None = None,
    filename: str = "informe_trafico",
) -> bytes:
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()
    counts = stats["counts"]
    cong_label, cong_color = congestion

    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(14, 9), facecolor="#16213e")
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        # Imagen anotada
        ax_img = fig.add_subplot(gs[0, :2])
        ax_img.imshow(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
        ax_img.axis("off")
        ax_img.set_title("Imagen analizada", color="white", fontsize=12, fontweight="bold")

        # Métricas numéricas
        ax_met = fig.add_subplot(gs[0, 2])
        ax_met.set_facecolor("#1a1a2e")
        ax_met.axis("off")
        lines = [
            ("Vehículos",  str(stats["vehicle_count"])),
            ("Peatones",   str(stats["person_count"])),
            ("Total",      str(stats["total_count"])),
            ("Densidad",   f"{density:.0f}%"),
            ("Estado",     cong_label),
        ]
        for k, (label, val) in enumerate(lines):
            ax_met.text(0.05, 0.88 - k * 0.18, label,  color="#aaa",   fontsize=10, transform=ax_met.transAxes)
            ax_met.text(0.95, 0.88 - k * 0.18, val,    color="white",  fontsize=13,
                        fontweight="bold", ha="right", transform=ax_met.transAxes)
        ax_met.set_title("Resumen", color="white", fontsize=12, fontweight="bold")

        # Gráfico de barras
        ax_bar = fig.add_subplot(gs[1, :2])
        ax_bar.set_facecolor("#1a1a2e")
        classes = [c for c in CLASS_NAMES if counts.get(c, 0) > 0]
        values  = [counts[c] for c in classes]
        colors  = [CLASS_COLORS_HEX.get(c, "#888") for c in classes]
        bars = ax_bar.bar(classes, values, color=colors, edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, values):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        str(v), ha="center", color="white", fontsize=10)
        ax_bar.set_facecolor("#1a1a2e")
        ax_bar.tick_params(colors="white")
        ax_bar.set_title("Distribución por clase", color="white", fontsize=11, fontweight="bold")
        for sp in ["top", "right"]:
            ax_bar.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]:
            ax_bar.spines[sp].set_color("#444")

        # Pie
        ax_pie = fig.add_subplot(gs[1, 2])
        ax_pie.set_facecolor("#1a1a2e")
        if values:
            wedge_colors = [CLASS_COLORS_HEX.get(c, "#888") for c in classes]
            ax_pie.pie(values, labels=classes, colors=wedge_colors,
                       autopct="%1.0f%%", textprops={"color": "white", "fontsize": 8})
        ax_pie.set_title("Proporción", color="white", fontsize=11, fontweight="bold")

        plt.suptitle("Smart Traffic Analyzer — Informe de análisis",
                     color="white", fontsize=14, fontweight="bold", y=1.01)
        pdf.savefig(fig, bbox_inches="tight", facecolor="#16213e")
        plt.close(fig)

        df = build_detection_dataframe(detections)
        if not df.empty:
            fig2, ax2 = plt.subplots(figsize=(14, max(4, len(df) * 0.45 + 1.5)),
                                      facecolor="#16213e")
            ax2.axis("off")
            tbl = ax2.table(
                cellText=df.values,
                colLabels=df.columns,
                cellLoc="center",
                loc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1.1, 1.5)
            for (row, col), cell in tbl.get_celld().items():
                cell.set_facecolor("#1a1a2e" if row > 0 else "#2c2c4e")
                cell.set_text_props(color="white")
                cell.set_edgecolor("#444")
            ax2.set_title("Detalle de detecciones", color="white", fontsize=13,
                          fontweight="bold", pad=20)
            pdf.savefig(fig2, bbox_inches="tight", facecolor="#16213e")
            plt.close(fig2)

        if signs:
            fig3, ax3 = plt.subplots(figsize=(14, max(4, len(signs) * 1.4 + 2)),
                                      facecolor="#16213e")
            ax3.axis("off")
            rows_s = []
            for i, s in enumerate(signs, 1):
                text = s.get("ocr_text", "") or "—"
                conf_str = f"{s['ocr_conf']:.0%}" if s.get("ocr_conf", 0) > 0 else "—"
                rows_s.append([str(i), s.get("emoji", ""), s.get("label", ""),
                                text[:60], conf_str, s.get("meaning", "")[:60]])
            if rows_s:
                tbl3 = ax3.table(
                    cellText=rows_s,
                    colLabels=["#", "", "Tipo", "Texto leído", "Confianza", "Significado"],
                    cellLoc="left", loc="center",
                )
                tbl3.auto_set_font_size(False)
                tbl3.set_fontsize(8)
                tbl3.scale(1.1, 1.6)
                for (row, col), cell in tbl3.get_celld().items():
                    cell.set_facecolor("#1a1a2e" if row > 0 else "#2c2c4e")
                    cell.set_text_props(color="white")
                    cell.set_edgecolor("#444")
            ax3.set_title("Señales de tráfico detectadas", color="white",
                          fontsize=13, fontweight="bold", pad=20)
            pdf.savefig(fig3, bbox_inches="tight", facecolor="#16213e")
            plt.close(fig3)

        if heatmap_bgr is not None:
            fig3, ax3 = plt.subplots(figsize=(10, 7), facecolor="#16213e")
            ax3.imshow(cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB))
            ax3.axis("off")
            ax3.set_title("Mapa de calor — concentración de vehículos",
                          color="white", fontsize=12, fontweight="bold")
            pdf.savefig(fig3, bbox_inches="tight", facecolor="#16213e")
            plt.close(fig3)

    buf.seek(0)
    return buf.read()


def generate_csv(detections: list[dict]) -> bytes:
    df = build_detection_dataframe(detections)
    return df.to_csv(index=False).encode("utf-8")
