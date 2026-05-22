#!/bin/bash
# Setup del entorno para Smart Traffic Analyzer

set -e

echo "=== Smart Traffic Analyzer — Setup ==="
echo ""

# Verificar Python
python3 --version || { echo "ERROR: Python 3 no encontrado."; exit 1; }

# Instalar dependencias
echo "Instalando dependencias..."
pip3 install -r requirements.txt

echo ""
echo "=== Setup completado ==="
echo ""
echo "Para ejecutar el pipeline completo:"
echo "  python3 run_pipeline.py"
echo ""
echo "Para lanzar solo la app:"
echo "  python3 run_pipeline.py --app"
echo ""
echo "Para ir directo a la app sin entrenar:"
echo "  python3 run_pipeline.py --app  (usa modelo pre-entrenado automáticamente)"
