"""
Script para realizar inferencias/predicciones rápidas
Útil para usar modelos ya entrenados
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO
import argparse


class ObjectDetectorFasterRCNN:
    """Detector usando Faster R-CNN"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        
        # Modificar classifier para 2 clases
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image_path, confidence_threshold=0.5):
        """Predicción en una imagen"""
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        # Filtrar por confidencia
        mask = scores > confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        
        return boxes, scores, image
    
    def draw_results(self, image, boxes, scores, output_path=None):
        """Dibujar resultados en imagen"""
        result_image = image.copy()
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f'Human: {score:.2f}'
            cv2.putText(result_image, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(str(output_path), result_image)
        
        return result_image


class ObjectDetectorYOLO:
    """Detector usando YOLOv8"""
    
    def __init__(self, model_path, device=None):
        if device is None:
            device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.device = device
    
    def predict(self, image_path, confidence_threshold=0.5):
        """Predicción en una imagen"""
        results = self.model.predict(
            source=str(image_path),
            conf=confidence_threshold,
            device=self.device
        )
        
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        
        image = cv2.imread(str(image_path))
        
        return boxes, scores, image
    
    def draw_results(self, image, boxes, scores, output_path=None):
        """Dibujar resultados"""
        result_image = image.copy()
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f'Human: {score:.2f}'
            cv2.putText(result_image, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(str(output_path), result_image)
        
        return result_image


def predict_single_image(image_path, model_type='yolo', model_path=None, confidence=0.5):
    """Predicción en una imagen individual"""
    
    print(f"Cargando imagen: {image_path}")
    
    if model_type.lower() == 'yolo':
        detector = ObjectDetectorYOLO(model_path or 'yolo_results/human_detection/weights/best.pt')
    else:
        detector = ObjectDetectorFasterRCNN(model_path or 'checkpoints/best_model_epoch_1.pth')
    
    print(f"Realizando predicción...")
    boxes, scores, image = detector.predict(image_path, confidence)
    
    print(f"Detecciones encontradas: {len(boxes)}")
    for i, (box, score) in enumerate(zip(boxes, scores)):
        print(f"  {i+1}. Score: {score:.4f}, Caja: {[int(v) for v in box]}")
    
    # Mostrar resultado
    result_image = detector.draw_results(image, boxes, scores)
    
    output_path = Path("./predictions") / "result.jpg"
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), result_image)
    
    print(f"✓ Resultado guardado: {output_path}")
    
    return boxes, scores


def predict_directory(dir_path, model_type='yolo', model_path=None, confidence=0.5):
    """Predicción en un directorio de imágenes"""
    
    image_files = list(Path(dir_path).glob('*.jpg')) + \
                  list(Path(dir_path).glob('*.jpeg')) + \
                  list(Path(dir_path).glob('*.png'))
    
    print(f"Encontradas {len(image_files)} imágenes")
    
    if model_type.lower() == 'yolo':
        detector = ObjectDetectorYOLO(model_path or 'yolo_results/human_detection/weights/best.pt')
    else:
        detector = ObjectDetectorFasterRCNN(model_path or 'checkpoints/best_model_epoch_1.pth')
    
    output_dir = Path("./predictions") / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_detections = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Procesando: {image_path.name}")
        
        try:
            boxes, scores, image = detector.predict(str(image_path), confidence)
            total_detections += len(boxes)
            
            print(f"  Detecciones: {len(boxes)}")
            
            result_image = detector.draw_results(image, boxes, scores)
            output_path = output_dir / f"{image_path.stem}_result.jpg"
            cv2.imwrite(str(output_path), result_image)
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\n✓ Predicciones guardadas en: {output_dir}")
    print(f"Total de detecciones: {total_detections}")


def main():
    parser = argparse.ArgumentParser(
        description='Realizar predicciones con modelos de detección de objetos'
    )
    
    parser.add_argument('--source', type=str, required=True,
                       help='Ruta a imagen o directorio')
    parser.add_argument('--model-type', type=str, choices=['yolo', 'fasterrcnn'],
                       default='yolo', help='Tipo de modelo')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Ruta al modelo entrenado')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Threshold de confianza')
    parser.add_argument('--output', type=str, default='./predictions',
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    source_path = Path(args.source)
    
    if source_path.is_file():
        print("Predicción en imagen individual")
        print("="*60)
        predict_single_image(
            source_path,
            model_type=args.model_type,
            model_path=args.model_path,
            confidence=args.confidence
        )
    
    elif source_path.is_dir():
        print("Predicción en directorio")
        print("="*60)
        predict_directory(
            source_path,
            model_type=args.model_type,
            model_path=args.model_path,
            confidence=args.confidence
        )
    
    else:
        print(f"Error: {source_path} no existe")


if __name__ == "__main__":
    """
    Ejemplos de uso:
    
    # Predicción en imagen individual (YOLOv8)
    python predict.py --source imagen.jpg --model-type yolo
    
    # Predicción en imagen individual (Faster R-CNN)
    python predict.py --source imagen.jpg --model-type fasterrcnn
    
    # Predicción en directorio
    python predict.py --source ./test_images --model-type yolo
    
    # Con modelo personalizado
    python predict.py --source imagen.jpg --model-type yolo \\
        --model-path ./yolo_results/human_detection/weights/best.pt
    
    # Con threshold custom
    python predict.py --source imagen.jpg --confidence 0.7
    """
    
    main()
