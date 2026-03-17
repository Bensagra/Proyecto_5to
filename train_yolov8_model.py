"""
Alternativa moderna: Entrenamiento con YOLOv8
Más rápido y eficiente que Faster R-CNN
"""

import torch
from pathlib import Path
import pandas as pd
import yaml
import os

# Instalar: pip install ultralytics opencv-python pandas

def create_yolo_dataset_config():
    """Crear configuración del dataset para YOLO"""
    dataset_path = Path("/Users/bensagra/Documents/Proyecto_5to/Human Dataset v2.v6-experiment-subject-4.tensorflow")
    
    # Crear estructura de directorios
    yolo_data_dir = Path("./yolo_dataset")
    yolo_data_dir.mkdir(exist_ok=True)
    
    for split in ["train", "valid", "test"]:
        (yolo_data_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_data_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Convertir anotaciones CSV a formato YOLO
    def csv_to_yolo(csv_file, img_dir, split):
        annotations = pd.read_csv(csv_file)
        
        for img_file in annotations['filename'].unique():
            # Copiar imagen
            src_img = img_dir / img_file
            dst_img = yolo_data_dir / "images" / split / img_file
            
            if src_img.exists():
                import shutil
                shutil.copy(src_img, dst_img)
                
                # Crear archivo label YOLO
                img_annotations = annotations[annotations['filename'] == img_file]
                img_width = img_annotations.iloc[0]['width']
                img_height = img_annotations.iloc[0]['height']
                
                label_file = yolo_data_dir / "labels" / split / (img_file.rsplit('.', 1)[0] + ".txt")
                
                with open(label_file, 'w') as f:
                    for _, row in img_annotations.iterrows():
                        # Convertir a formato YOLO normalizado (center_x, center_y, width, height, class)
                        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                        
                        center_x = (xmin + xmax) / (2 * img_width)
                        center_y = (ymin + ymax) / (2 * img_height)
                        width = (xmax - xmin) / img_width
                        height = (ymax - ymin) / img_height
                        
                        class_id = 0  # Human
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    # Procesar cada split
    print("Convirtiendo anotaciones a formato YOLO...")
    csv_to_yolo(dataset_path / "train" / "_annotations.csv", 
                dataset_path / "train", "train")
    csv_to_yolo(dataset_path / "valid" / "_annotations.csv", 
                dataset_path / "valid", "valid")
    csv_to_yolo(dataset_path / "test" / "_annotations.csv", 
                dataset_path / "test", "test")
    
    # Crear archivo data.yaml
    data_yaml = {
        'path': str(yolo_data_dir.absolute()),
        'train': 'images/train',
        'val': 'images/valid',
        'test': 'images/test',
        'nc': 1,  # número de clases
        'names': ['Human']
    }
    
    with open(yolo_data_dir / "data.yaml", 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"✓ Dataset YOLO creado en: {yolo_data_dir}")
    return yolo_data_dir / "data.yaml"


def train_yolov8():
    """Entrenar modelo YOLOv8"""
    from ultralytics import YOLO
    
    print("="*60)
    print("ENTRENAMIENTO CON YOLOv8")
    print("="*60)
    
    # Crear configuración del dataset
    data_yaml = create_yolo_dataset_config()
    
    # Crear modelo
    print("\nCargando YOLOv8 nano (más rápido)...")
    model = YOLO('yolov8n.pt')  # nano
    # Alternativas: yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large)
    
    # Entrenar
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    results = model.train(
        data=str(data_yaml),
        epochs=20,
        imgsz=416,
        batch=8,
        device=device,
        patience=10,
        save=True,
        project='./yolo_results',
        name='human_detection',
        pretrained=True,
        optimizer='Adam',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        fl_gamma=0.0,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        quad=False,
        fraction=1.0,
        seed=0,
        verbose=True,
        amp=True,
        multi_scale=False,
        plots=True,
    )
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60)
    
    return model, results


def test_yolov8_model():
    """Probar el modelo entrenado"""
    from ultralytics import YOLO
    import cv2
    import numpy as np
    
    print("\n" + "="*60)
    print("PREDICCIÓN CON MODELO ENTRENADO")
    print("="*60)
    
    # Cargar el modelo mejor entrenado
    model = YOLO('./yolo_results/human_detection/weights/best.pt')
    
    # Seleccionar una imagen de test
    test_dir = Path("/Users/bensagra/Documents/Proyecto_5to/Human Dataset v2.v6-experiment-subject-4.tensorflow/test")
    test_images = list(test_dir.glob("*.jpg"))
    
    if test_images:
        test_image = str(test_images[0])
        print(f"\nUsando imagen: {Path(test_image).name}")
        
        # Predicción
        results = model.predict(
            source=test_image,
            conf=0.5,  # confidence threshold
            iou=0.45,  # IOU threshold
            save=True,
            project='./predictions',
            name='human_detection',
            device=0 if torch.cuda.is_available() else 'cpu'
        )
        
        # Mostrar resultados
        for result in results:
            print(f"\nDetecciones encontradas: {len(result.boxes)}")
            for i, box in enumerate(result.boxes):
                print(f"  {i+1}. Confidencia: {box.conf.item():.4f}")
                print(f"     Coordenadas: {box.xyxy.cpu().numpy()[0].astype(int)}")


def predict_batch(image_dir, output_dir='./predictions'):
    """Predicción en un lote de imágenes"""
    from ultralytics import YOLO
    
    model = YOLO('./yolo_results/human_detection/weights/best.pt')
    
    results = model.predict(
        source=str(image_dir),
        conf=0.5,
        iou=0.45,
        save=True,
        project=output_dir,
        device=0 if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"✓ Predicciones guardadas en: {output_dir}")
    
    # Estadísticas
    total_detections = sum(len(r.boxes) for r in results)
    print(f"Total de detecciones: {total_detections}")


if __name__ == "__main__":
    # Entrenar
    model, results = train_yolov8()
    
    # Test
    test_yolov8_model()
    
    # Predicción en batch
    test_dir = Path("/Users/bensagra/Documents/Proyecto_5to/Human Dataset v2.v6-experiment-subject-4.tensorflow/test")
    predict_batch(test_dir)
