import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# =======================
# CONFIGURACIÓN
# =======================
DATASET_PATH = Path("/Users/bensagra/Documents/Proyecto_5to/Human Dataset v2.v6-experiment-subject-4.tensorflow")
TRAIN_DIR = DATASET_PATH / "train"
VAL_DIR = DATASET_PATH / "valid"
TEST_DIR = DATASET_PATH / "test"

BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# =======================
# DATASET CUSTOM
# =======================
class HumanDetectionDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        
        # Leer CSV de anotaciones
        self.annotations = pd.read_csv(annotations_file)
        self.image_files = self.annotations['filename'].unique()
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self.img_dir / img_name
        
        # Cargar imagen
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: No se pudo cargar {img_path}")
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        # Obtener anotaciones para esta imagen
        img_annotations = self.annotations[self.annotations['filename'] == img_name]
        
        boxes = []
        labels = []
        
        for _, row in img_annotations.iterrows():
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # clase 1 = Human
        
        if len(boxes) == 0:
            boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
            labels = np.array([0], dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        
        # Convertir a tensores
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        target = {
            'boxes': torch.from_numpy(boxes),
            'labels': torch.from_numpy(labels)
        }
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, target


# =======================
# FUNCIONES DE ENTRENAMIENTO
# =======================
def collate_fn(batch):
    """Custom collate para Faster R-CNN"""
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None
    return tuple(zip(*batch))


def train_one_epoch(model, train_loader, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0
    epoch_start = time.time()
    
    # Barra de progreso
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                desc=f'Epoch {epoch}/{num_epochs} [TRAIN]', 
                leave=True, ncols=100)
    
    batch_losses = []
    
    for batch_idx, (images, targets) in pbar:
        if images is None:
            continue
            
        batch_start = time.time()
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        batch_loss = losses.item()
        total_loss += batch_loss
        batch_losses.append(batch_loss)
        
        batch_time = time.time() - batch_start
        
        # Actualizar barra de progreso con información
        avg_loss = np.mean(batch_losses[-10:])  # Promedio de últimos 10 batches
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'avg_loss': f'{avg_loss:.4f}',
            'time/batch': f'{batch_time:.2f}s'
        })
    
    pbar.close()
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / len(train_loader)
    
    print(f"  ✓ Entrenamiento completado en {epoch_time:.1f}s")
    print(f"  📊 Loss promedio: {avg_loss:.4f}")
    
    return avg_loss


@torch.no_grad()
def evaluate(model, val_loader, device, epoch, num_epochs):
    model.eval()
    total_loss = 0
    epoch_start = time.time()
    
    # Barra de progreso para validación
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), 
                desc=f'Epoch {epoch}/{num_epochs} [VAL] ', 
                leave=True, ncols=100)
    
    batch_losses = []
    
    for batch_idx, (images, targets) in pbar:
        if images is None:
            continue
            
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        batch_loss = losses.item()
        
        total_loss += batch_loss
        batch_losses.append(batch_loss)
        
        avg_loss = np.mean(batch_losses[-10:])
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'avg_loss': f'{avg_loss:.4f}',
        })
    
    pbar.close()
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / len(val_loader)
    
    print(f"  ✓ Validación completada en {epoch_time:.1f}s")
    print(f"  📊 Validation loss: {avg_loss:.4f}")
    
    return avg_loss


def train_model():
    print("\n" + "="*80)
    print("🤖 ENTRENAMIENTO DE MODELO DE DETECCIÓN DE OBJETOS".center(80))
    print("="*80)
    print(f"📍 Device: {DEVICE}".ljust(40) + f"📦 Batch Size: {BATCH_SIZE}")
    print(f"🔄 Epochs: {NUM_EPOCHS}".ljust(40) + f"📚 Learning Rate: {LEARNING_RATE}")
    print("="*80 + "\n")
    
    # Crear datasets
    print("📂 Cargando datasets...")
    start_load = time.time()
    
    train_dataset = HumanDetectionDataset(
        TRAIN_DIR, 
        DATASET_PATH / "train" / "_annotations.csv"
    )
    val_dataset = HumanDetectionDataset(
        VAL_DIR, 
        DATASET_PATH / "valid" / "_annotations.csv"
    )
    
    load_time = time.time() - start_load
    print(f"  ✓ Train samples: {len(train_dataset):,}")
    print(f"  ✓ Val samples: {len(val_dataset):,}")
    print(f"  ⏱️  Tiempo de carga: {load_time:.2f}s\n")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Modelo Faster R-CNN
    print("🧠 Cargando modelo Faster R-CNN...")
    model_load_start = time.time()
    
    # Cargar modelo pretrained con pesos COCO (91 clases)
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Modificar el classifier para nuestro número de clases (2: background + human)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    
    model.to(DEVICE)
    
    model_load_time = time.time() - model_load_start
    print(f"  ✓ Modelo cargado en {model_load_time:.2f}s\n")
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Entrenamiento
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    total_training_start = time.time()
    
    print("🚀 Iniciando entrenamiento...\n")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        print(f"\n{'='*80}")
        print(f"🔹 EPOCH {epoch}/{NUM_EPOCHS}".ljust(40) + f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}".rjust(40))
        print(f"{'='*80}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch, NUM_EPOCHS)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = evaluate(model, val_loader, DEVICE, epoch, NUM_EPOCHS)
        val_losses.append(val_loss)
        
        # Learning rate scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Información de progreso
        improvement = ""
        if val_loss < best_loss:
            improvement = " ⭐ MEJOR MODELO"
            best_loss = val_loss
            checkpoint_path = CHECKPOINT_DIR / f"best_model_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"\n  ✅ Mejor modelo guardado: {checkpoint_path.name}")
        
        # Estadísticas de la epoch
        loss_diff = val_loss - train_loss
        print(f"\n  📈 Epoch Time: {epoch_time:.1f}s")
        print(f"  📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Diff: {loss_diff:.4f}{improvement}\n")
        
        # Estimación de tiempo restante
        avg_epoch_time = (time.time() - total_training_start) / epoch
        remaining_epochs = NUM_EPOCHS - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_minutes = eta_seconds / 60
        
        if remaining_epochs > 0:
            print(f"  ⏱️  ETA: {eta_minutes:.1f} minutos ({remaining_epochs} epochs restantes)")
    
    total_training_time = time.time() - total_training_start
    
    # Guardar modelo final
    final_model_path = CHECKPOINT_DIR / "final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Modelo final guardado: {final_model_path}")
    
    # Resumen final
    print(f"\n{'='*80}")
    print("📊 RESUMEN DEL ENTRENAMIENTO".center(80))
    print(f"{'='*80}")
    print(f"⏱️  Tiempo total: {total_training_time/60:.2f} minutos ({total_training_time/3600:.2f} horas)")
    print(f"🏆 Best validation loss: {best_loss:.4f}")
    print(f"📉 Mejor epoch: {np.argmin(val_losses) + 1}")
    print(f"📈 Final train loss: {train_losses[-1]:.4f}")
    print(f"📈 Final val loss: {val_losses[-1]:.4f}")
    print(f"📁 Checkpoints guardados en: {CHECKPOINT_DIR.absolute()}")
    print(f"{'='*80}\n")
    
    # Generar gráficos
    print("📊 Generando gráficos...")
    plot_training_history(train_losses, val_losses)
    
    return model, train_losses, val_losses


def plot_training_history(train_losses, val_losses):
    """Generar gráficos de historias de entrenamiento"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Pérdida
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    axes[0].plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(epochs)
    
    # Gráfico 2: Mejora relativa
    improvement = [(train_losses[i] - val_losses[i]) / train_losses[i] * 100 
                   for i in range(len(train_losses))]
    colors = ['green' if x > 0 else 'red' for x in improvement]
    axes[1].bar(epochs, improvement, color=colors, alpha=0.7, edgecolor='black')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Val vs Train Loss Improvement', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticks(epochs)
    
    plt.tight_layout()
    
    # Guardar figura
    plot_path = CHECKPOINT_DIR / "training_history.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Gráfico guardado: {plot_path}")
    plt.close()
    
    # Mostrar estadísticas en gráfico
    print(f"\n  📊 Estadísticas:")
    print(f"     • Pérdida mínima: {min(val_losses):.4f}")
    print(f"     • Pérdida máxima: {max(val_losses):.4f}")
    print(f"     • Mejora promedio: {np.mean(improvement):.2f}%")


def predict_on_image(model, image_path, device, confidence_threshold=0.5):
    """Realizar predicciones en una imagen"""
    model.eval()
    
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    # Filtrar por confidencia
    mask = scores > confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    
    # Dibujar resultados
    image_with_boxes = image.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_with_boxes, f'{score:.2f}', (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image_with_boxes, boxes, scores


if __name__ == "__main__":
    # Entrenar modelo
    model, train_losses, val_losses = train_model()
    
    # Opcional: Predicción en una imagen de test
    print("\n" + "="*60)
    print("REALIZANDO PREDICCIÓN EN IMAGEN DE TEST")
    print("="*60)
    
    test_images = list(TEST_DIR.glob("*.jpg"))
    if test_images:
        test_image_path = test_images[0]
        print(f"\nUsando imagen: {test_image_path.name}")
        
        model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        
        best_checkpoint = max(CHECKPOINT_DIR.glob("best_model_epoch_*.pth"), 
                             key=lambda p: p.stat().st_mtime)
        checkpoint = torch.load(best_checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        
        result_image, boxes, scores = predict_on_image(model, test_image_path, DEVICE)
        
        # Guardar resultado
        output_path = CHECKPOINT_DIR / "prediction_result.jpg"
        cv2.imwrite(str(output_path), result_image)
        print(f"✓ Resultado guardado: {output_path}")
        print(f"Detecciones: {len(boxes)}")
        for i, (box, score) in enumerate(zip(boxes, scores)):
            print(f"  {i+1}. Score: {score:.4f}, Box: {box.astype(int)}")
