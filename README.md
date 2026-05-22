# Computer Vision Projects

Este repositorio contiene tres proyectos de computer vision desarrollados como parte del trabajo de fin de curso.

## Proyectos

- **golf_cv** — Detección y análisis de imágenes en el contexto del golf
- **semantic_segmentation** — Segmentación semántica de imágenes
- **trafic** — Detección y análisis de tráfico

Cada proyecto tiene su propio `README.md` con instrucciones detalladas de uso.

---

## Descarga de datos

Las imágenes y datasets de entrenamiento no están incluidos en este repositorio por su tamaño. Descárgalos desde el siguiente enlace y coloca cada carpeta dentro del proyecto correspondiente, manteniendo la estructura original:

📁 [Descargar datos desde Google Drive](https://drive.google.com/drive/folders/1kMA3Kogb6fw-qCoSZ7f163XJ6SxKlIpI?usp=sharing)

### Estructura esperada tras la descarga

```
computer vision/
├── golf_cv/
│   └── data/
│       ├── patches/
│       ├── raw/
│       └── results/
├── semantic_segmentation/
│   └── data/
│       ├── raw/
│       └── samples/
└── trafic/
    ├── data/
    ├── samples/
    ├── results/
    └── models/
        └── traffic_detector/
```

---

## Instalación

1. Clona el repositorio
2. Descarga los datos desde el enlace de arriba
3. Instala las dependencias de cada proyecto:

```bash
pip install -r requirements.txt
```

4. Sigue las instrucciones del `README.md` de cada proyecto para ejecutarlo
