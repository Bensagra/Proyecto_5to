#!/bin/bash
# Script de inicio rápido para entrenar modelo de detección

echo "🚀 INICIO RÁPIDO - ENTRENAMIENTO DE DETECCIÓN DE OBJETOS"
echo "========================================================"
echo ""

# Comprobar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 no está instalado"
    exit 1
fi

echo "✓ Python3 encontrado"
echo ""

# Crear entorno virtual
echo "📦 Creando entorno virtual..."
python3 -m venv venv
source venv/bin/activate

echo "✓ Entorno virtual activado"
echo ""

# Instalar dependencias
echo "📥 Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✓ Dependencias instaladas"
echo ""

# Seleccionar modelo
echo "🎯 ELIGE OPCIÓN:"
echo ""
echo "1) YOLOv8 (RECOMENDADO - Más rápido y eficiente)"
echo "2) Faster R-CNN (Más potente pero lento)"
echo ""
read -p "Opción (1 o 2): " choice

echo ""
echo "========================================================"
echo "INICIANDO ENTRENAMIENTO"
echo "========================================================"
echo ""

if [ "$choice" == "1" ] || [ "$choice" == "" ]; then
    echo "🔧 Entrenando con YOLOv8..."
    python train_yolov8_model.py
elif [ "$choice" == "2" ]; then
    echo "🔧 Entrenando con Faster R-CNN..."
    python train_detection_model.py
else
    echo "❌ Opción inválida"
    exit 1
fi

echo ""
echo "========================================================"
echo "✓ ¡ENTRENAMIENTO COMPLETADO!"
echo "========================================================"
echo ""
echo "📊 Próximos pasos:"
echo ""
echo "1. Revisar resultados en:"
echo "   - yolo_results/ (si usaste YOLOv8)"
echo "   - checkpoints/ (si usaste Faster R-CNN)"
echo ""
echo "2. Hacer predicciones:"
echo "   python predict.py --source <imagen.jpg>"
echo ""
echo "3. Consultar guía completa:"
echo "   cat TRAINING_GUIDE.md"
echo ""
