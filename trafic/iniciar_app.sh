#!/bin/bash
# Arranque limpio de la app — mata procesos viejos y limpia caché

echo "Parando procesos anteriores..."
pkill -f "streamlit" 2>/dev/null
pkill -f "app.py" 2>/dev/null
sleep 1

echo "Limpiando caché Python..."
find . -name "*.pyc" -delete 2>/dev/null
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

echo "Iniciando Smart Traffic Analyzer..."
cd "$(dirname "$0")"
streamlit run stage5_application/app.py
