"""
Genera un PDF completo con toda la documentación del proyecto de segmentación semántica.
Uso: python3 generar_pdf.py
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfgen import canvas

BASE_DIR = Path(__file__).resolve().parent
OUT_PDF = BASE_DIR / "outputs" / "documentacion_proyecto.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

# ── Paleta de colores ──────────────────────────────────────────────────────
C_DARK    = colors.HexColor("#1e293b")
C_BLUE    = colors.HexColor("#3b82f6")
C_LIGHT   = colors.HexColor("#f1f5f9")
C_ACCENT  = colors.HexColor("#0ea5e9")
C_GREEN   = colors.HexColor("#22c55e")
C_ORANGE  = colors.HexColor("#f97316")
C_PURPLE  = colors.HexColor("#8b5cf6")
C_GRAY    = colors.HexColor("#64748b")
C_BORDER  = colors.HexColor("#e2e8f0")
C_WHITE   = colors.white
C_CODE_BG = colors.HexColor("#f8fafc")


# ── Estilos tipográficos ───────────────────────────────────────────────────
def build_styles():
    base = getSampleStyleSheet()

    styles = {
        "cover_title": ParagraphStyle("cover_title",
            fontSize=32, leading=40, textColor=C_WHITE,
            fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=8),
        "cover_sub": ParagraphStyle("cover_sub",
            fontSize=15, leading=22, textColor=colors.HexColor("#cbd5e1"),
            fontName="Helvetica", alignment=TA_CENTER, spaceAfter=6),
        "cover_badge": ParagraphStyle("cover_badge",
            fontSize=11, leading=16, textColor=colors.HexColor("#94a3b8"),
            fontName="Helvetica", alignment=TA_CENTER),

        "h1": ParagraphStyle("h1",
            fontSize=20, leading=26, textColor=C_DARK,
            fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=6,
            borderPad=0),
        "h2": ParagraphStyle("h2",
            fontSize=14, leading=20, textColor=C_BLUE,
            fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4),
        "h3": ParagraphStyle("h3",
            fontSize=11, leading=16, textColor=C_DARK,
            fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=3),

        "body": ParagraphStyle("body",
            fontSize=10, leading=16, textColor=C_DARK,
            fontName="Helvetica", alignment=TA_JUSTIFY, spaceAfter=6),
        "bullet": ParagraphStyle("bullet",
            fontSize=10, leading=15, textColor=C_DARK,
            fontName="Helvetica", leftIndent=16, spaceAfter=3,
            bulletIndent=6, bulletFontName="Helvetica"),
        "code": ParagraphStyle("code",
            fontSize=8.5, leading=13, textColor=colors.HexColor("#334155"),
            fontName="Courier", leftIndent=12, spaceAfter=2,
            backColor=C_CODE_BG),
        "caption": ParagraphStyle("caption",
            fontSize=9, leading=13, textColor=C_GRAY,
            fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceAfter=4),
        "label": ParagraphStyle("label",
            fontSize=9, leading=13, textColor=C_GRAY,
            fontName="Helvetica"),
        "toc_h1": ParagraphStyle("toc_h1",
            fontSize=11, leading=16, textColor=C_DARK,
            fontName="Helvetica-Bold", spaceBefore=4),
        "toc_h2": ParagraphStyle("toc_h2",
            fontSize=10, leading=14, textColor=C_GRAY,
            fontName="Helvetica", leftIndent=14),
    }
    return styles


S = build_styles()


# ── Helpers ───────────────────────────────────────────────────────────────
def hr(color=C_BORDER, thickness=0.8):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=4, spaceBefore=4)


def sp(h=0.3):
    return Spacer(1, h * cm)


def p(text, style="body"):
    return Paragraph(text, S[style])


def bullet(text):
    return Paragraph(f"• {text}", S["bullet"])


def code(text):
    return Paragraph(text, S["code"])


def section_header(number, title, color=C_BLUE):
    data = [[
        Paragraph(f"<font color='white'><b>{number}</b></font>", ParagraphStyle(
            "sh_num", fontSize=13, leading=16, fontName="Helvetica-Bold",
            textColor=C_WHITE, alignment=TA_CENTER)),
        Paragraph(f"<b>{title}</b>", ParagraphStyle(
            "sh_title", fontSize=14, leading=18, fontName="Helvetica-Bold",
            textColor=C_WHITE)),
    ]]
    t = Table(data, colWidths=[1.2 * cm, None])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [color]),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (0, 0), 10),
        ("LEFTPADDING", (1, 0), (1, 0), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("ROUNDEDCORNERS", [6]),
    ]))
    return t


def info_box(text, color=C_LIGHT, border=C_BLUE):
    data = [[Paragraph(text, ParagraphStyle(
        "ib", fontSize=9.5, leading=15, fontName="Helvetica",
        textColor=C_DARK))]]
    t = Table(data, colWidths=[17 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEBEFORE", (0, 0), (0, -1), 3, border),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return t


def two_col_table(rows, headers=None, col_widths=None):
    """Generic styled table."""
    if col_widths is None:
        col_widths = [5 * cm, 12 * cm]
    data = []
    if headers:
        data.append([Paragraph(f"<b>{h}</b>", S["label"]) for h in headers])
    for row in rows:
        data.append([Paragraph(str(c), S["body"]) for c in row])

    t = Table(data, colWidths=col_widths, repeatRows=1 if headers else 0)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), C_DARK if headers else C_LIGHT),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE if headers else C_DARK),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    t.setStyle(TableStyle(style))
    return t


# ── Numeración de páginas ─────────────────────────────────────────────────
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_page_number(num_pages)
            super().showPage()
        super().save()

    def _draw_page_number(self, page_count):
        if self._pageNumber > 1:
            self.setFont("Helvetica", 8)
            self.setFillColor(C_GRAY)
            self.drawRightString(
                A4[0] - 1.5 * cm, 1 * cm,
                f"Página {self._pageNumber} de {page_count}"
            )
            self.drawString(1.5 * cm, 1 * cm, "Segmentación Semántica — ADE20K")


# ── Portada ───────────────────────────────────────────────────────────────
def cover_page():
    elements = []

    # Fondo oscuro simulado con tabla full-width
    title_block = [
        [Paragraph("Segmentación Semántica<br/>de Imágenes", S["cover_title"])],
        [Paragraph("Sistema completo de Computer Vision sobre ADE20K (150 clases)", S["cover_sub"])],
        [Spacer(1, 0.5 * cm)],
        [Paragraph("SegFormer-B0 · PyTorch · HuggingFace · Gradio", S["cover_badge"])],
        [Spacer(1, 0.3 * cm)],
        [Paragraph("Universidad · Proyecto Final de Computer Vision", S["cover_badge"])],
    ]
    t = Table([[row[0]] for row in title_block], colWidths=[17 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_DARK),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
        ("ROUNDEDCORNERS", [10]),
    ]))
    elements.append(Spacer(1, 3.5 * cm))
    elements.append(t)
    elements.append(Spacer(1, 1.5 * cm))

    # Tarjetas de métricas en portada
    cards = [
        ["Dataset", "ADE20K\nscene_parse_150"],
        ["Clases", "150\nsemánticas"],
        ["Modelo", "SegFormer-B0\n(NVIDIA)"],
        ["Framework", "PyTorch\n+ HuggingFace"],
    ]
    card_data = [[
        Paragraph(f"<b><font color='#64748b'>{k}</font></b><br/>"
                  f"<font size='13'><b>{v.replace(chr(10),'<br/>')}</b></font>",
                  ParagraphStyle("card", fontSize=10, leading=16,
                                 fontName="Helvetica-Bold", textColor=C_DARK,
                                 alignment=TA_CENTER))
        for k, v in cards
    ]]
    card_table = Table(card_data, colWidths=[4.2 * cm] * 4)
    card_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_LIGHT),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("GRID", (0, 0), (-1, -1), 1, C_BORDER),
        ("ROUNDEDCORNERS", [8]),
    ]))
    elements.append(card_table)
    elements.append(Spacer(1, 2 * cm))

    elements.append(hr(C_BORDER))
    elements.append(p(
        "Este documento recoge el diseño, implementación, tecnologías y decisiones técnicas "
        "del sistema de segmentación semántica construido como proyecto universitario de "
        "Computer Vision. El pipeline cubre desde la descarga automática del dataset hasta "
        "la aplicación web interactiva con interfaz Gradio.",
        "body"
    ))

    elements.append(PageBreak())
    return elements


# ── Índice ────────────────────────────────────────────────────────────────
def toc_section():
    elements = [p("Índice de contenidos", "h1"), hr(), sp(0.4)]
    entries = [
        ("1.", "Descripción general del proyecto", "h1"),
        ("2.", "Stack tecnológico", "h1"),
        ("3.", "Arquitectura del modelo — SegFormer-B0", "h1"),
        ("4.", "Estructura de carpetas", "h1"),
        ("5.", "Etapa 1 — Adquisición de datos", "h1"),
        ("   5.1", "Dataset ADE20K", "h2"),
        ("   5.2", "Proceso de descarga y organización", "h2"),
        ("6.", "Etapa 2 — Preprocesado y aumentación", "h1"),
        ("   6.1", "Pipeline de transformaciones", "h2"),
        ("   6.2", "Filtros y aumentaciones (albumentations)", "h2"),
        ("   6.3", "SegmentationDataset y DataLoaders", "h2"),
        ("7.", "Etapa 3 — Entrenamiento", "h1"),
        ("   7.1", "Fine-tuning de SegFormer-B0", "h2"),
        ("   7.2", "Función de pérdida y optimizador", "h2"),
        ("   7.3", "Métricas: mIoU", "h2"),
        ("8.", "Etapa 4 — Inferencia", "h1"),
        ("   8.1", "Pipeline de inferencia", "h2"),
        ("   8.2", "Mapa de colores y visualización", "h2"),
        ("9.", "Etapa 5 — Aplicación Gradio", "h1"),
        ("10.", "Tests automáticos", "h1"),
        ("11.", "Guía de ejecución", "h1"),
    ]
    for num, title, level in entries:
        style = S["toc_h1"] if level == "h1" else S["toc_h2"]
        elements.append(Paragraph(f"{num}  {title}", style))
    elements.append(PageBreak())
    return elements


# ── Sección 1: Descripción general ────────────────────────────────────────
def section_overview():
    elements = [section_header("1", "Descripción general del proyecto"), sp()]
    elements.append(p(
        "El proyecto construye un <b>sistema completo de segmentación semántica</b> que, "
        "dada cualquier imagen fotográfica, produce una versión coloreada donde cada "
        "categoría de objeto (cielo, coches, personas, edificios, vegetación, etc.) "
        "aparece pintada con un color distinto. El resultado es un mapa denso de etiquetas "
        "semánticas que asigna una clase a cada píxel de la imagen."
    ))
    elements.append(p(
        "El sistema recorre todas las fases del pipeline de Computer Vision moderno: "
        "obtención automática de datos, preprocesado, fine-tuning de un modelo "
        "transformer pre-entrenado, inferencia sobre imágenes nuevas y despliegue como "
        "aplicación web interactiva."
    ))
    elements.append(sp(0.4))
    elements.append(info_box(
        "<b>Objetivo académico:</b> demostrar el dominio del pipeline completo de Computer "
        "Vision aplicado a segmentación semántica, incluyendo uso de datasets reales, "
        "modelos state-of-the-art y evaluación cuantitativa con métricas estándar (mIoU).",
        color=colors.HexColor("#eff6ff"), border=C_BLUE
    ))
    elements.append(sp(0.5))

    elements.append(p("El sistema se divide en cinco etapas independientes:", "h3"))
    etapas = [
        ["Etapa 1", "Adquisición de datos", "Descarga automática de ADE20K desde HuggingFace (3 700 imágenes)"],
        ["Etapa 2", "Preprocesado", "Normalización, aumentación con albumentations, DataLoaders"],
        ["Etapa 3", "Entrenamiento", "Fine-tuning de SegFormer-B0, 10 épocas, métricas mIoU"],
        ["Etapa 4", "Inferencia", "Segmentación de imágenes nuevas, overlay coloreado"],
        ["Etapa 5", "Aplicación", "Interfaz web Gradio con leyenda de clases y estadísticas"],
    ]
    elements.append(two_col_table(
        etapas,
        headers=["Etapa", "Nombre", "Descripción"],
        col_widths=[2.5 * cm, 4 * cm, 10.5 * cm]
    ))
    elements.append(PageBreak())
    return elements


# ── Sección 2: Stack tecnológico ─────────────────────────────────────────
def section_stack():
    elements = [section_header("2", "Stack tecnológico", C_PURPLE), sp()]
    elements.append(p(
        "El proyecto usa exclusivamente herramientas open-source del ecosistema Python "
        "para Computer Vision y Deep Learning. A continuación se detalla cada librería, "
        "su versión mínima y su rol concreto en el sistema."
    ))
    elements.append(sp(0.3))

    libs = [
        ["PyTorch ≥ 2.0", "Framework de Deep Learning", "Red neuronal, entrenamiento, backpropagation, inferencia GPU/MPS/CPU"],
        ["torchvision ≥ 0.15", "Utilidades de visión", "Transformaciones de tensor, utilidades de imagen"],
        ["transformers ≥ 4.35", "Modelos HuggingFace", "Carga de SegFormerForSemanticSegmentation pre-entrenado"],
        ["datasets ≥ 2.14", "Datasets HuggingFace", "Descarga y streaming de ADE20K (scene_parse_150)"],
        ["albumentations ≥ 1.3", "Aumentación de imágenes", "Pipeline de transformaciones: flip, brillo, rotación, blur, HSV"],
        ["opencv-python ≥ 4.8", "Procesado de imagen", "Lectura/escritura de imágenes, operaciones morfológicas"],
        ["Pillow ≥ 10.0", "Manipulación de imágenes", "Carga de PNG/JPEG, conversión de modos de color"],
        ["matplotlib ≥ 3.7", "Visualización", "Gráficas de entrenamiento, mapas de color, grid de resultados"],
        ["seaborn ≥ 0.12", "Visualización estadística", "Histogramas de distribución de clases"],
        ["Gradio ≥ 4.0", "Interfaz web", "App interactiva con subida de imagen, tabs, slider de opacidad"],
        ["tqdm ≥ 4.65", "Barras de progreso", "Seguimiento visual de entrenamiento e inferencia"],
        ["numpy ≥ 1.24", "Álgebra numérica", "Operaciones sobre arrays de imágenes y máscaras"],
        ["scikit-learn ≥ 1.3", "Utilidades ML", "Class weights, métricas auxiliares"],
        ["pandas ≥ 2.0", "Tablas de datos", "Estadísticas de clases en la app Gradio"],
        ["pytest ≥ 7.4", "Testing", "Suite de tests automáticos por etapa"],
        ["requests ≥ 2.31", "Peticiones HTTP", "Descarga de imágenes de internet en tests"],
    ]
    elements.append(two_col_table(
        libs,
        headers=["Librería", "Categoría", "Uso en el proyecto"],
        col_widths=[4 * cm, 3.5 * cm, 9.5 * cm]
    ))
    elements.append(PageBreak())
    return elements


# ── Sección 3: Arquitectura SegFormer ─────────────────────────────────────
def section_architecture():
    elements = [section_header("3", "Arquitectura del modelo — SegFormer-B0", C_GREEN), sp()]

    elements.append(p(
        "<b>SegFormer</b> es una arquitectura transformer para segmentación semántica "
        "propuesta por NVIDIA en 2021. Elimina la necesidad de decodificadores complejos "
        "(como los de U-Net o DeepLab) combinando un encoder jerárquico Mix Transformer "
        "(MiT) con un decodificador MLP ligero."
    ))
    elements.append(sp(0.3))

    elements.append(p("Componentes principales:", "h3"))
    components = [
        ["Mix Transformer Encoder (MiT-B0)",
         "Encoder jerárquico de 4 stages con Self-Attention de eficiencia lineal. "
         "Produce feature maps a escalas 1/4, 1/8, 1/16 y 1/32 de la imagen de entrada. "
         "La variante B0 tiene ~3.7M parámetros (la más ligera de la familia)."],
        ["Efficient Self-Attention",
         "Reduce la complejidad cuadrática del attention estándar mediante un ratio de "
         "reducción de secuencia (R). En lugar de calcular atención sobre todos los tokens, "
         "se reduce la dimensión de K y V por un factor configurable."],
        ["Mix-FFN",
         "Feed-forward network que incorpora una convolución 3×3 de paso cero entre "
         "las capas lineales, introduciendo información posicional local sin necesitar "
         "embeddings de posición explícitos."],
        ["All-MLP Decoder",
         "Decodificador minimalista que: (1) alinea los canales de los 4 stages a una "
         "dimensión común con proyecciones lineales, (2) hace upsample bilineal al "
         "tamaño 1/4, (3) concatena y fusiona con una MLP, (4) predice las C=150 "
         "clases con una convolución 1×1."],
    ]
    elements.append(two_col_table(
        components,
        headers=["Componente", "Descripción"],
        col_widths=[5.5 * cm, 11.5 * cm]
    ))
    elements.append(sp(0.5))

    elements.append(p("Especificaciones técnicas:", "h3"))
    specs = [
        ["Variante usada", "SegFormer-B0 (más eficiente, adecuada para hardware limitado)"],
        ["Checkpoint base", "nvidia/segformer-b0-finetuned-ade-512-512 (HuggingFace Hub)"],
        ["Parámetros totales", "~3.7M (encoder) + ~0.4M (decoder) = ~4.1M parámetros"],
        ["Resolución entrada", "512 × 512 píxeles (RGB normalizado con stats ImageNet)"],
        ["Resolución salida", "128 × 128 logits → interpolación bilineal a 512 × 512"],
        ["Clases de salida", "150 clases semánticas (ADE20K scene_parse_150)"],
        ["Pre-entrenamiento", "ImageNet-1K + fine-tuning ADE20K por NVIDIA"],
    ]
    elements.append(two_col_table(specs, col_widths=[5.5 * cm, 11.5 * cm]))
    elements.append(sp(0.5))

    elements.append(info_box(
        "<b>¿Por qué SegFormer-B0?</b> Es el balance perfecto para este proyecto: "
        "suficientemente preciso en ADE20K (mIoU ~37% sin fine-tuning propio) y lo "
        "bastante ligero para correr en CPU/Apple Silicon sin necesitar GPU NVIDIA. "
        "Además, HuggingFace ofrece el checkpoint ya fine-tuneado en ADE20K, lo que "
        "permite obtener resultados de calidad incluso sin reentrenar.",
        color=colors.HexColor("#f0fdf4"), border=C_GREEN
    ))
    elements.append(PageBreak())
    return elements


# ── Sección 4: Estructura de carpetas ─────────────────────────────────────
def section_structure():
    elements = [section_header("4", "Estructura de carpetas", C_ORANGE), sp()]
    elements.append(p(
        "El proyecto sigue una organización modular donde cada directorio tiene un "
        "propósito único y las dependencias entre módulos son unidireccionales "
        "(etapa N solo importa de etapas anteriores)."
    ))
    elements.append(sp(0.4))

    tree_lines = [
        ("semantic_segmentation/", C_DARK, True),
        ("├── data/", C_BLUE, False),
        ("│   ├── raw/", C_GRAY, False),
        ("│   │   ├── train/         img_000000.jpg + mask_000000.png  × 3 000", C_GRAY, False),
        ("│   │   ├── validation/    × 500 pares imagen/máscara", C_GRAY, False),
        ("│   │   ├── test/          × 200 pares imagen/máscara", C_GRAY, False),
        ("│   │   └── metadata.json  Resumen del dataset (splits, clases, resolución)", C_GRAY, False),
        ("│   ├── processed/", C_BLUE, False),
        ("│   │   └── dataset_stats.json  Media/std por canal, pixels/clase, pesos", C_GRAY, False),
        ("│   └── samples/           10 imágenes representativas para demos rápidas", C_GRAY, False),
        ("├── models/", C_BLUE, False),
        ("│   └── checkpoints/", C_GRAY, False),
        ("│       ├── best_model.pth       Mejor checkpoint (mayor mIoU en val)", C_GRAY, False),
        ("│       ├── last_model.pth       Checkpoint de la última época", C_GRAY, False),
        ("│       └── training_history.json  Curvas de loss y mIoU por época", C_GRAY, False),
        ("├── outputs/", C_BLUE, False),
        ("│   ├── test_results/", C_GRAY, False),
        ("│   │   ├── overlay_0000.jpg … overlay_0019.jpg  (20 overlays de test)", C_GRAY, False),
        ("│   │   └── internet_samples/   Resultados sobre imágenes de internet", C_GRAY, False),
        ("│   └── plots/", C_GRAY, False),
        ("│       ├── etapa1_exploracion.png       Pares imagen/máscara + histograma", C_GRAY, False),
        ("│       ├── etapa2_augmentations.png     Misma imagen con 6 aumentaciones", C_GRAY, False),
        ("│       ├── etapa3_training_curves.png   Loss y mIoU por época", C_GRAY, False),
        ("│       ├── etapa4_resultados.png        Grid 4×3 de resultados", C_GRAY, False),
        ("│       └── etapa4_metricas_por_clase.png  Barplot IoU por clase", C_GRAY, False),
        ("├── src/", C_BLUE, False),
        ("│   ├── etapa1_datos.py", C_DARK, False),
        ("│   ├── etapa2_preprocesado.py", C_DARK, False),
        ("│   ├── etapa3_entrenamiento.py", C_DARK, False),
        ("│   ├── etapa4_inferencia.py", C_DARK, False),
        ("│   └── etapa5_app.py", C_DARK, False),
        ("├── tests/", C_BLUE, False),
        ("│   ├── conftest.py", C_GRAY, False),
        ("│   ├── test_etapa1.py", C_GRAY, False),
        ("│   ├── test_etapa2.py", C_GRAY, False),
        ("│   ├── test_etapa3.py", C_GRAY, False),
        ("│   └── test_etapa4.py", C_GRAY, False),
        ("├── run_pipeline.py         Script maestro que ejecuta todo en orden", C_DARK, False),
        ("├── generar_pdf.py          Genera esta documentación", C_DARK, False),
        ("├── requirements.txt", C_GRAY, False),
        ("└── README.md", C_GRAY, False),
    ]

    tree_data = []
    for line, color, bold in tree_lines:
        font = "Courier-Bold" if bold else "Courier"
        style = ParagraphStyle("tree", fontSize=8, leading=12, fontName=font,
                               textColor=color, leftIndent=4)
        tree_data.append([Paragraph(line, style)])

    tree_table = Table(tree_data, colWidths=[17 * cm])
    tree_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_CODE_BG),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("BOX", (0, 0), (-1, -1), 1, C_BORDER),
        ("ROUNDEDCORNERS", [6]),
    ]))
    elements.append(tree_table)
    elements.append(PageBreak())
    return elements


# ── Sección 5: Etapa 1 ────────────────────────────────────────────────────
def section_etapa1():
    elements = [section_header("5", "Etapa 1 — Adquisición de datos", C_BLUE), sp()]

    elements.append(p("5.1  Dataset ADE20K (scene_parse_150)", "h2"))
    elements.append(p(
        "<b>ADE20K</b> es uno de los benchmarks de referencia en segmentación semántica. "
        "La variante <i>scene_parse_150</i> contiene imágenes de escenas de interior y "
        "exterior con anotaciones pixel-a-pixel para 150 categorías semánticas."
    ))
    ds_info = [
        ["Fuente", "MIT CSAIL — distribuido vía HuggingFace Hub"],
        ["Identificador HF", "scene_parse_150"],
        ["Total imágenes", "~25 574 (train) + 2 000 (val)"],
        ["Imágenes usadas", "3 000 (train) + 500 (val) + 200 (test)"],
        ["Clases", "150 categorías semánticas"],
        ["Formato imagen", "JPEG RGB, resolución variable (~400-800px)"],
        ["Formato máscara", "PNG uint8, valores enteros 0-149 (índice de clase)"],
        ["Clase 0", "wall (pared) — clase más frecuente"],
        ["Clase 2", "sky (cielo) — clase usada como referencia en tests"],
    ]
    elements.append(two_col_table(ds_info, col_widths=[4.5 * cm, 12.5 * cm]))
    elements.append(sp(0.5))

    elements.append(p("5.2  Proceso de descarga y organización", "h2"))
    elements.append(p(
        "La descarga es automática y reproducible mediante la API de HuggingFace Datasets. "
        "No es necesario descargar manualmente ningún archivo."
    ))
    steps = [
        "Se llama a <b>load_dataset('scene_parse_150')</b> que descarga y cachea el dataset.",
        "Se itera sobre los splits train y validation, guardando pares imagen/máscara con nombres consecutivos: <b>img_000001.jpg</b> y <b>mask_000001.png</b>.",
        "El split de test se obtiene de los primeros 200 elementos de validation (el dataset no tiene split de test separado).",
        "Se copian 10 imágenes representativas a <b>data/samples/</b> para demos rápidas.",
        "Se genera <b>data/raw/metadata.json</b> con estadísticas del dataset.",
        "Se genera la figura <b>etapa1_exploracion.png</b> con pares imagen/máscara y distribución de clases.",
    ]
    for s in steps:
        elements.append(bullet(s))
    elements.append(sp(0.5))

    elements.append(info_box(
        "<b>Distribución de clases:</b> ADE20K es un dataset muy desbalanceado. Las clases "
        "'wall', 'building', 'sky' y 'floor' dominan en número de píxeles, mientras que "
        "clases como 'microwave', 'flag' o 'crt screen' tienen muy pocos píxeles. "
        "Este desbalance se compensa en la Etapa 3 con una <b>CrossEntropyLoss ponderada</b>.",
        color=colors.HexColor("#fffbeb"), border=C_ORANGE
    ))
    elements.append(PageBreak())
    return elements


# ── Sección 6: Etapa 2 ────────────────────────────────────────────────────
def section_etapa2():
    elements = [section_header("6", "Etapa 2 — Preprocesado y aumentación", C_PURPLE), sp()]

    elements.append(p("6.1  Pipeline de transformaciones", "h2"))
    elements.append(p(
        "Se definen tres pipelines distintos con <b>albumentations</b>, cada uno adaptado "
        "al split correspondiente. La normalización sigue las estadísticas de ImageNet "
        "porque el encoder MiT-B0 fue pre-entrenado sobre ImageNet."
    ))
    elements.append(sp(0.3))

    pipelines = [
        ["train_transform",
         "Resize(512,512) → HorizontalFlip(p=0.5) → RandomBrightnessContrast(p=0.3) "
         "→ RandomRotate90(p=0.2) → Normalize(ImageNet) → ToTensorV2",
         "Aumentación aleatoria para mejorar generalización"],
        ["val_transform",
         "Resize(512,512) → Normalize(ImageNet) → ToTensorV2",
         "Sin aumentación; evaluación determinista"],
        ["test_transform",
         "Igual que val_transform",
         "Idéntico al de validación para comparar métricas"],
    ]
    elements.append(two_col_table(
        pipelines,
        headers=["Pipeline", "Transformaciones", "Propósito"],
        col_widths=[3.5 * cm, 9.5 * cm, 4 * cm]
    ))
    elements.append(sp(0.5))

    elements.append(p("6.2  Filtros y aumentaciones — albumentations", "h2"))
    elements.append(p(
        "Todas las transformaciones de albumentations se aplican de forma <b>conjunta y "
        "consistente a imagen y máscara</b>: si la imagen se voltea horizontalmente, "
        "la máscara se voltea de la misma manera, preservando la correspondencia píxel a píxel."
    ))
    elements.append(sp(0.3))

    augmentations = [
        ["Resize(512, 512)",
         "Redimensiona imagen y máscara a 512×512. Usa interpolación bilineal para la "
         "imagen e interpolación por vecino más cercano para la máscara (preserva valores enteros)."],
        ["HorizontalFlip(p=0.5)",
         "Volteo horizontal aleatorio con probabilidad 50%. Simula escenas vistas desde "
         "el espejo. Es la aumentación más efectiva para escenas de interior/exterior."],
        ["RandomBrightnessContrast(p=0.3)",
         "Altera aleatoriamente el brillo y contraste de la imagen en un rango de ±30%. "
         "Ayuda al modelo a ser robusto a diferentes condiciones de iluminación."],
        ["RandomRotate90(p=0.2)",
         "Rotación aleatoria en múltiplos de 90° con probabilidad 20%. Útil para "
         "escenas donde la orientación puede variar."],
        ["Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
         "Normalización estándar de ImageNet. Convierte píxeles de [0,255] a floats "
         "normalizados alrededor de cero con desviación ~1. Obligatorio para modelos "
         "pre-entrenados en ImageNet."],
        ["ToTensorV2()",
         "Convierte arrays NumPy (H,W,C) a tensores PyTorch (C,H,W). Transpone "
         "los canales al formato esperado por PyTorch sin copiar datos (operación O(1))."],
        ["GaussianBlur (solo plot demo)",
         "Desenfoque gaussiano con kernel variable. Se usa en la figura de demostración "
         "de aumentaciones pero no en el pipeline de entrenamiento."],
        ["HueSaturationValue (solo plot demo)",
         "Altera tono, saturación y valor en el espacio HSV. Simula diferentes "
         "condiciones de color de la cámara. Visible en la figura etapa2_augmentations.png."],
        ["ShiftScaleRotate (solo plot demo)",
         "Desplazamiento, escala y rotación arbitraria combinados. Útil para datasets "
         "pequeños pero no incluido en el pipeline final por riesgo de distorsión excesiva."],
    ]
    elements.append(two_col_table(
        augmentations,
        headers=["Transformación", "Descripción técnica"],
        col_widths=[5.5 * cm, 11.5 * cm]
    ))
    elements.append(sp(0.5))

    elements.append(p("6.3  SegmentationDataset y DataLoaders", "h2"))
    elements.append(p(
        "La clase <b>SegmentationDataset</b> (hereda de torch.utils.data.Dataset) gestiona "
        "la carga de pares imagen/máscara y la aplicación de transformaciones."
    ))
    impl_details = [
        ["Carga de imagen", "PIL.Image + conversión a NumPy RGB (H,W,3 uint8)"],
        ["Carga de máscara", "PIL.Image → NumPy uint8 → clip [0,149] → int64"],
        ["Aplicación de transforms", "albumentations Compose recibe image= y mask= simultáneamente"],
        ["Valores de retorno", "Tensor imagen (3,512,512) float32 + Tensor máscara (512,512) long"],
        ["DataLoader train", "batch=8, shuffle=True, num_workers=2, pin_memory=True, drop_last=True"],
        ["DataLoader val/test", "batch=4, shuffle=False, num_workers=2, pin_memory=True"],
    ]
    elements.append(two_col_table(impl_details, col_widths=[5 * cm, 12 * cm]))
    elements.append(PageBreak())
    return elements


# ── Sección 7: Etapa 3 ────────────────────────────────────────────────────
def section_etapa3():
    elements = [section_header("7", "Etapa 3 — Entrenamiento", C_GREEN), sp()]

    elements.append(p("7.1  Fine-tuning de SegFormer-B0", "h2"))
    elements.append(p(
        "En lugar de entrenar desde cero, se aplica <b>fine-tuning</b> sobre el checkpoint "
        "nvidia/segformer-b0-finetuned-ade-512-512. Esto permite partir de un modelo ya "
        "optimizado para ADE20K y adaptar ligeramente los pesos con los datos propios."
    ))
    elements.append(p(
        "Se usa el parámetro <code>ignore_mismatched_sizes=True</code> de HuggingFace para "
        "permitir reemplazar la cabeza de clasificación si el número de clases difiere."
    ))
    elements.append(sp(0.4))

    config = [
        ["Modelo base", "nvidia/segformer-b0-finetuned-ade-512-512"],
        ["Épocas", "10"],
        ["Learning rate inicial", "6e-5 (AdamW)"],
        ["Weight decay", "0.01"],
        ["Scheduler", "CosineAnnealingLR (T_max=10, eta_min=1e-6)"],
        ["Gradient clipping", "max_norm=1.0 (evita explosión de gradientes)"],
        ["Batch size train", "8 imágenes por iteración"],
        ["Batch size val", "4 imágenes por iteración"],
    ]
    elements.append(two_col_table(config, col_widths=[5.5 * cm, 11.5 * cm]))
    elements.append(sp(0.5))

    elements.append(p("7.2  Función de pérdida y optimizador", "h2"))
    elements.append(p(
        "<b>CrossEntropyLoss ponderada:</b> la pérdida estándar para clasificación "
        "multiclase por píxel. Se pondera con los class_weights calculados en la Etapa 2 "
        "para compensar el desbalance de clases. Las clases más raras reciben un peso "
        "mayor (inversamente proporcional a su frecuencia, capped a ×10)."
    ))
    elements.append(p(
        "<b>AdamW:</b> variante de Adam con weight decay desacoplado. Es el optimizador "
        "estándar para transformers. El weight decay de 0.01 actúa como regularización L2 "
        "aplicada directamente a los pesos (no al momento del gradiente)."
    ))
    elements.append(p(
        "<b>CosineAnnealingLR:</b> el learning rate decrece siguiendo una curva coseno "
        "desde 6e-5 hasta 1e-6. Esto evita oscilaciones al final del entrenamiento y "
        "favorece la convergencia en mínimos más planos."
    ))
    elements.append(sp(0.4))

    elements.append(p("7.3  Métricas: mIoU", "h2"))
    elements.append(p(
        "<b>mIoU (mean Intersection over Union)</b> es la métrica estándar en segmentación "
        "semántica. Para cada clase <i>c</i>, el IoU mide la superposición entre la "
        "predicción y la ground truth:"
    ))
    elements.append(sp(0.2))
    elements.append(info_box(
        "IoU(c) = |predicción(c) ∩ groundtruth(c)| / |predicción(c) ∪ groundtruth(c)|"
        "          =  TP / (TP + FP + FN)\n\n"
        "mIoU = media de IoU(c) sobre todas las clases presentes en el dataset",
        color=colors.HexColor("#f0fdf4"), border=C_GREEN
    ))
    elements.append(sp(0.3))
    elements.append(p(
        "Un mIoU de 0.0 significa que no hay ninguna superposición (predicciones completamente "
        "erróneas). Un mIoU de 1.0 sería predicción perfecta píxel a píxel. "
        "SegFormer-B0 pre-entrenado alcanza ~37% mIoU en ADE20K sin fine-tuning adicional."
    ))
    elements.append(sp(0.4))

    elements.append(info_box(
        "<b>Guardado de checkpoints:</b> se guardan dos ficheros en models/checkpoints/. "
        "<b>best_model.pth</b> se sobreescribe cada vez que la mIoU de validación mejora. "
        "<b>last_model.pth</b> se sobreescribe al final de cada época. Ambos almacenan: "
        "state_dict del modelo, state_dict del optimizador, número de época, mIoU y loss.",
        color=colors.HexColor("#eff6ff"), border=C_BLUE
    ))
    elements.append(PageBreak())
    return elements


# ── Sección 8: Etapa 4 ────────────────────────────────────────────────────
def section_etapa4():
    elements = [section_header("8", "Etapa 4 — Inferencia", C_ACCENT), sp()]

    elements.append(p("8.1  Pipeline de inferencia", "h2"))
    elements.append(p(
        "La función central es <b>segmentar_imagen(ruta, modelo, device)</b>, que acepta "
        "cualquier imagen JPG/PNG de cualquier resolución y devuelve tres arrays NumPy: "
        "la imagen original, la máscara coloreada y el overlay blended."
    ))
    elements.append(sp(0.3))

    pipeline = [
        ["1. Carga", "PIL.Image.open() + conversión RGB → NumPy (H, W, 3)"],
        ["2. Preproceso", "Resize(512,512) + Normalize ImageNet + ToTensorV2 → tensor (1,3,512,512)"],
        ["3. Inferencia", "modelo(pixel_values=tensor) → logits (1, 150, 128, 128)"],
        ["4. Upsampling", "F.interpolate(logits, size=(H_orig, W_orig), mode='bilinear') → (1,150,H,W)"],
        ["5. Argmax", "logits.argmax(dim=1) → máscara de índices (H, W) con valores [0,149]"],
        ["6. Colorización", "COLOR_ARRAY[pred_mask] → imagen RGB (H, W, 3) usando indexación vectorizada"],
        ["7. Overlay", "alpha * colored_mask + (1-alpha) * original → imagen blended"],
    ]
    elements.append(two_col_table(
        pipeline,
        headers=["Paso", "Operación"],
        col_widths=[2.5 * cm, 14.5 * cm]
    ))
    elements.append(sp(0.5))

    elements.append(p("8.2  Mapa de colores y visualización", "h2"))
    elements.append(p(
        "El <b>COLOR_MAP</b> es una lista de 150 colores RGB construida concatenando "
        "varios colormaps de matplotlib (<i>tab20, tab20b, tab20c, Set1, Set2, Set3, "
        "Paired, Accent</i>) para maximizar la distinción visual entre clases adyacentes."
    ))
    elements.append(p(
        "La asignación color → clase es determinista y fija: la clase 0 (wall) siempre "
        "recibe el mismo color, facilitando la interpretación visual repetida. "
        "Se usa indexación NumPy vectorizada (<code>COLOR_ARRAY[pred_mask]</code>) para "
        "generar la imagen coloreada en una sola operación O(H×W)."
    ))
    elements.append(sp(0.3))

    elements.append(p("Evaluación en test set:", "h3"))
    eval_steps = [
        "Se procesa el split completo de test (200 imágenes) y se acumulan predicciones y ground truths.",
        "Se calcula el <b>mIoU global</b> como media sobre todas las clases presentes.",
        "Se calcula el <b>mIoU por clase</b> (150 valores) para identificar qué categorías predicen mejor/peor.",
        "Se mide la <b>velocidad de inferencia</b> en imágenes/segundo.",
        "Los primeros 20 overlays se guardan en <b>outputs/test_results/</b>.",
        "Se generan los plots <b>etapa4_resultados.png</b> y <b>etapa4_metricas_por_clase.png</b>.",
    ]
    for s in eval_steps:
        elements.append(bullet(s))
    elements.append(PageBreak())
    return elements


# ── Sección 9: Etapa 5 ────────────────────────────────────────────────────
def section_etapa5():
    elements = [section_header("9", "Etapa 5 — Aplicación Gradio", C_ORANGE), sp()]

    elements.append(p(
        "La etapa final envuelve el sistema en una <b>aplicación web interactiva</b> "
        "construida con Gradio. Cualquier persona puede usarla sin tocar el código: "
        "solo subir una imagen y obtener la segmentación al instante."
    ))
    elements.append(sp(0.3))

    features = [
        ["Subida de imagen", "Acepta JPG y PNG por drag & drop, click o portapapeles"],
        ["Pestaña Segmentación", "Máscara coloreada pura (150 colores, uno por clase)"],
        ["Pestaña Overlay", "Imagen original + segmentación blended con opacidad ajustable"],
        ["Slider de opacidad", "Control deslizante 0-100% para mezclar original con máscara"],
        ["Tarjetas resumen", "Clases detectadas, clase dominante, tiempo de inferencia, dispositivo"],
        ["Leyenda de colores", "Chips de color con nombre de clase y % de píxeles, orden descendente"],
        ["Galería de ejemplos", "6 imágenes pre-cargadas de data/samples/ para demo rápida"],
        ["Lanzamiento", "python3 src/etapa5_app.py → abre http://localhost:7860 automáticamente"],
        ["Modelo lazy-load", "El modelo se carga en memoria solo la primera vez que se hace una predicción"],
        ["Fallback de pesos", "Si no existe best_model.pth, usa los pesos pre-entrenados de HuggingFace"],
    ]
    elements.append(two_col_table(
        features,
        headers=["Característica", "Detalle"],
        col_widths=[5 * cm, 12 * cm]
    ))
    elements.append(sp(0.4))
    elements.append(info_box(
        "<b>Leyenda dinámica:</b> Los chips de la leyenda se generan en tiempo real para "
        "cada imagen. Solo aparecen las clases realmente presentes en la predicción, "
        "ordenadas de mayor a menor área. El color de texto de cada chip (negro o blanco) "
        "se elige automáticamente según la luminancia del color de fondo para maximizar legibilidad.",
        color=colors.HexColor("#fdf4ff"), border=C_PURPLE
    ))
    elements.append(PageBreak())
    return elements


# ── Sección 10: Tests ─────────────────────────────────────────────────────
def section_tests():
    elements = [section_header("10", "Tests automáticos", colors.HexColor("#ef4444")), sp()]

    elements.append(p(
        "El proyecto incluye una suite de tests con <b>pytest</b>, organizada por etapas. "
        "Los tests verifican tanto la corrección estructural (shapes, tipos, rangos de valores) "
        "como la corrección funcional (el modelo aprende, la inferencia cubre todos los píxeles)."
    ))
    elements.append(sp(0.3))

    tests_data = [
        ["test_etapa1.py", "test_minimum_train_images", "≥ 2 500 imágenes en train"],
        ["", "test_minimum_val_images", "≥ 400 imágenes en validation"],
        ["", "test_image_mask_pairs_exist", "Cada imagen tiene su máscara (100 muestras)"],
        ["", "test_mask_values_in_range", "Máscaras en [0, 149]"],
        ["", "test_images_are_valid_rgb", "Imágenes válidas modo RGB"],
        ["", "test_metadata_file_exists", "metadata.json existe y tiene 150 clases"],
        ["test_etapa2.py", "test_train_batch_shape", "Shape batch = (8, 3, 512, 512)"],
        ["", "test_val_batch_shape", "Shape batch val = (4, 3, 512, 512)"],
        ["", "test_images_normalized", "Media ≈ 0, std razonable tras normalización"],
        ["", "test_mask_values_valid", "Máscaras en [0, 149] tras preproceso"],
        ["", "test_dataset_returns_tensors", "Dataset devuelve Tensor float32 + long"],
        ["", "test_dataset_stats_file_exists", "dataset_stats.json con 150 class weights"],
        ["test_etapa3.py", "test_best_model_checkpoint_exists", "best_model.pth existe"],
        ["", "test_model_output_shape", "Salida modelo = (B, 150, H/4, W/4)"],
        ["", "test_forward_pass_no_error", "Forward pass sin errores ni NaN"],
        ["", "test_model_improved_miou", "mIoU final > mIoU inicial"],
        ["", "test_predictions_are_valid_class_indices", "Predicciones en [0, 149]"],
        ["test_etapa4.py", "test_inference_*_no_error", "Inferencia sobre 3 imágenes de internet sin error"],
        ["", "test_masks_cover_all_pixels", "100% píxeles cubiertos (sin píxeles sin clase)"],
        ["", "test_sky_detected_in_landscape", "Clase 'sky' presente en imagen de paisaje"],
        ["", "test_colored_mask_shape_matches_original", "Máscara coloreada = mismas dimensiones que original"],
        ["", "test_output_images_saved", "Resultados guardados en internet_samples/"],
    ]

    elements.append(two_col_table(
        tests_data,
        headers=["Fichero", "Test", "Qué verifica"],
        col_widths=[3.5 * cm, 6.5 * cm, 7 * cm]
    ))
    elements.append(sp(0.4))
    elements.append(p("Para ejecutar todos los tests:", "h3"))
    elements.append(code("$ pytest tests/ -v --tb=short"))
    elements.append(sp(0.2))
    elements.append(p("Para ejecutar una etapa concreta:", "h3"))
    elements.append(code("$ pytest tests/test_etapa1.py -v"))
    elements.append(PageBreak())
    return elements


# ── Sección 11: Guía de ejecución ─────────────────────────────────────────
def section_execution():
    elements = [section_header("11", "Guía de ejecución", C_DARK), sp()]

    elements.append(p("Requisitos previos:", "h3"))
    elements.append(bullet("Python 3.10 o superior (testeado con 3.10–3.14)"))
    elements.append(bullet("Conexión a internet para la descarga del dataset y el modelo"))
    elements.append(bullet("~4 GB de espacio libre en disco (dataset + modelo + outputs)"))
    elements.append(bullet("8 GB RAM mínimo; GPU/Apple Silicon MPS recomendable para entrenamiento"))
    elements.append(sp(0.4))

    elements.append(p("Instalación de dependencias:", "h3"))
    for line in [
        "$ cd semantic_segmentation/",
        "$ pip3 install -r requirements.txt",
    ]:
        elements.append(code(line))
    elements.append(sp(0.4))

    options = [
        ["Pipeline completo", "python3 run_pipeline.py",
         "Ejecuta etapas 1→5 en orden con tests entre etapas. ~3-4h en CPU, ~45min con GPU."],
        ["Solo la app (rápido)", "python3 src/etapa5_app.py",
         "Lanza directamente la app con pesos pre-entrenados. Funciona sin entrenar. ~2min."],
        ["Saltarse entrenamiento", "python3 run_pipeline.py --skip-train",
         "Descarga datos + preprocesa + infiere con checkpoint existente + lanza app."],
        ["Saltarse descarga", "python3 run_pipeline.py --skip-download",
         "Asume que data/raw/ ya existe. Solo preprocesa, entrena e infiere."],
        ["Etapa individual", "python3 src/etapaN_xxx.py",
         "Ejecuta una sola etapa. Útil para depurar o re-ejecutar partes concretas."],
        ["Tests", "pytest tests/ -v",
         "Ejecuta la suite completa de tests. Requiere las etapas previas completadas."],
        ["Generar este PDF", "python3 generar_pdf.py",
         "Genera outputs/documentacion_proyecto.pdf con toda la documentación."],
    ]
    elements.append(two_col_table(
        options,
        headers=["Acción", "Comando", "Descripción"],
        col_widths=[4 * cm, 5.5 * cm, 7.5 * cm]
    ))
    elements.append(sp(0.5))

    elements.append(info_box(
        "<b>Tip para el profesor:</b> el modo más rápido para probar el sistema es "
        "instalar los requisitos y ejecutar <b>python3 src/etapa5_app.py</b>. "
        "El modelo se descarga automáticamente de HuggingFace (~90 MB) y la app abre "
        "en el navegador en http://localhost:7860. Se puede subir cualquier imagen "
        "descargada de internet y obtener la segmentación en pocos segundos.",
        color=colors.HexColor("#f0fdf4"), border=C_GREEN
    ))
    return elements


# ── Construcción del PDF ───────────────────────────────────────────────────
def build_pdf():
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Segmentación Semántica — ADE20K",
        author="Proyecto Computer Vision",
        subject="Documentación técnica del sistema de segmentación semántica",
    )

    story = []
    story += cover_page()
    story += toc_section()
    story += section_overview()
    story += section_stack()
    story += section_architecture()
    story += section_structure()
    story += section_etapa1()
    story += section_etapa2()
    story += section_etapa3()
    story += section_etapa4()
    story += section_etapa5()
    story += section_tests()
    story += section_execution()

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"\nPDF generado: {OUT_PDF}")


if __name__ == "__main__":
    build_pdf()
