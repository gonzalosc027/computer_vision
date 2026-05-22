# Segmentación Semántica — ADE20K (150 clases)

Sistema completo de segmentación semántica basado en **SegFormer-B0** fine-tuneado sobre el dataset **ADE20K (scene_parse_150)**. Dado cualquier imagen, produce una versión coloreada donde cada categoría de objeto aparece con un color distinto.

---

## Estructura del proyecto

```
semantic_segmentation/
├── data/
│   ├── raw/              # Imágenes y máscaras originales (train/validation/test)
│   ├── processed/        # dataset_stats.json con estadísticas
│   └── samples/          # 10 imágenes representativas para pruebas rápidas
├── models/
│   └── checkpoints/      # best_model.pth, last_model.pth, training_history.json
├── outputs/
│   ├── test_results/     # Overlays de las 20 primeras imágenes de test
│   │   └── internet_samples/  # Resultados sobre imágenes de internet
│   └── plots/            # Curvas de loss, mIoU, exploración, augmentaciones
├── src/
│   ├── etapa1_datos.py          # Descarga ADE20K desde HuggingFace
│   ├── etapa2_preprocesado.py   # Dataset, DataLoaders, augmentaciones
│   ├── etapa3_entrenamiento.py  # Fine-tuning SegFormer-B0 (10 épocas)
│   ├── etapa4_inferencia.py     # Inferencia + métricas sobre test set
│   └── etapa5_app.py            # App Gradio interactiva
├── tests/
│   ├── test_etapa1.py
│   ├── test_etapa2.py
│   ├── test_etapa3.py
│   └── test_etapa4.py
├── run_pipeline.py       # Ejecuta todo el pipeline
└── requirements.txt
```

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Ejecución

### Pipeline completo (recomendado)
```bash
python run_pipeline.py
```

### Solo la aplicación (con modelo ya entrenado)
```bash
python run_pipeline.py --only-app
```

### Saltarse el entrenamiento (usar checkpoint existente)
```bash
python run_pipeline.py --skip-train
```

### Etapas individuales
```bash
python src/etapa1_datos.py       # Descargar dataset
python src/etapa2_preprocesado.py # Preprocesar
python src/etapa3_entrenamiento.py # Entrenar
python src/etapa4_inferencia.py   # Inferencia
python src/etapa5_app.py          # Lanzar app en localhost:7860
```

### Tests
```bash
pytest tests/ -v
```

---

## Stack tecnológico

| Componente | Librería |
|---|---|
| Framework DL | PyTorch + torchvision |
| Modelo | HuggingFace Transformers (SegFormer-B0) |
| Dataset | HuggingFace Datasets (scene_parse_150) |
| Aumentaciones | albumentations |
| Visualización | matplotlib + seaborn |
| Interfaz web | Gradio |
| Preprocesado imagen | OpenCV + Pillow |

---

## Modelo

- **Arquitectura**: SegFormer-B0 (encoder jerárquico Mix Transformer + decoder MLP)
- **Pre-entrenamiento**: ADE20K (150 clases) — `nvidia/segformer-b0-finetuned-ade-512-512`
- **Fine-tuning**: 10 épocas, AdamW (lr=6e-5), CosineAnnealingLR, CrossEntropyLoss ponderada
- **Resolución de entrada**: 512×512

---

## Dataset

**ADE20K (scene_parse_150)** via HuggingFace:
- Train: 3.000 imágenes
- Validation: 500 imágenes  
- Test: 200 imágenes
- 150 categorías semánticas (wall, sky, building, person, car, tree, etc.)
