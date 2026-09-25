"""
CNN Disease Detection Training Script
Trains MobileNetV2 for plant disease classification using Kaggle dataset.
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.ai.disease_detection.model import PlantDiseaseMobileNetV2, save_model, create_model
from backend.ai.disease_detection.preprocess import (
    PreprocessingPipeline, get_train_augmentations, get_val_augmentations,
    IMAGENET_MEAN, IMAGENET_STD
)
from backend.ai.disease_detection.kaggle_dataset import KaggleDatasetManager, get_plant_disease_dataset
from backend.ai.disease_detection.dataset import PlantDiseaseDataset


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[Training] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"[Training] Using CPU")
    return device


class PlantDiseaseDataset(Dataset):
    """PyTorch Dataset for plant disease images."""
    
    def __init__(self, image_paths: list, labels: list, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from backend.ai.disease_detection.preprocess import preprocess_image_opencv
        
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Try OpenCV preprocessing
            img = preprocess_image_opencv(image_path)
            if img is None:
                # Fallback to PIL
                from backend.ai.disease_detection.preprocess import preprocess_image_pil
                img = preprocess_image_pil(image_path)
            
            if img is None:
                # Return a zero tensor if image can't be loaded
                img = torch.zeros(3, 224, 224)
            
            # Apply transform if provided
            if self.transform and isinstance(self.transform, transforms.Compose):
                # Convert tensor back to PIL for torchvision transforms
                img_pil = transforms.ToPILImage()(img.cpu())
                img = self.transform(img_pil)
            
            return img, label
            
        except Exception as e:
            print(f"[Dataset] Error loading {image_path}: {e}")
            return torch.zeros(3, 224, 224), label


class AlbumentationsDataset(Dataset):
    """Dataset using Albumentations transforms."""
    
    def __init__(self, image_paths: list, labels: list, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        import cv2
        from backend.ai.disease_detection.preprocess import apply_augmentation
        
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transform
        if self.transform:
            augmented = self.transform(image=img)
            img_tensor = augmented['image']
        else:
            # Basic preprocessing
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        
        return img_tensor, label


def train_model(data_dir: str,
                output_dir: str = "models",
                batch_size: int = 32,
                epochs: int = 10,
                learning_rate: float = 0.001,
                num_workers: int = 2,
                seed: int = 42,
                use_albumentations: bool = True,
                freeze_backbone: bool = False,
                early_stop_patience: int = 5,
                device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Train MobileNetV2 model on plant disease dataset.
    
    Args:
        data_dir: Path to dataset (extracted Kaggle dataset)
        output_dir: Directory to save model artifacts
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        num_workers: DataLoader workers
        seed: Random seed
        use_albumentations: Use albumentations for data augmentation
        freeze_backbone: Freeze pretrained backbone (train only classifier)
        early_stop_patience: Early stopping patience
        device: Torch device
    
    Returns:
        Training history dict
    """
    if device is None:
        device = get_device()
    
    # Set seeds
    set_seed(seed)
    
    # Setup paths
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "num_workers": num_workers,
        "seed": seed,
        "use_albumentations": use_albumentations,
        "freeze_backbone": freeze_backbone,
        "early_stop_patience": early_stop_patience,
        "device": str(device),
        "training_date": datetime.now().isoformat()
    }
    
    config_path = output_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"[Training] Configuration saved to: {config_path}")
    print(f"[Training] Device: {device}")
    print(f"[Training] Batch size: {batch_size}")
    print(f"[Training] Epochs: {epochs}")
    print(f"[Training] Learning rate: {learning_rate}")
    
    # Dataset preparation
    print(f"\n[Training] Loading dataset from: {data_dir}")
    dataset_analyzer = PlantDiseaseDataset(str(data_dir))
    class_names = dataset_analyzer.discover_classes()
    
    if len(class_names) == 0:
        print("[Training] ERROR: No classes found in dataset")
        return {"error": "No classes found"}
    
    # Build class mapping
    class_mapping = {i: name for i, name in enumerate(sorted(class_names))}
    reverse_mapping = {name: i for i, name in enumerate(sorted(class_names))}
    
    num_classes = len(class_names)
    print(f"[Training] Found {num_classes} classes: {sorted(class_names)}")
    
    # Split dataset into train/val/test
    # 80% train, 10% validation, 10% test
    all_images = []
    all_labels = []
    
    for class_idx, class_name in class_mapping.items():
        class_dir = data_dir / class_name
        if not class_dir.exists():
            class_dir = data_dir / class_name.replace('___', '_').replace('_', ' ')
        if not class_dir.exists():
            continue
        
        images = list(class_dir.glob('*.[jp][pn]g')) + list(class_dir.glob('*.jpg')) + \
                 list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
        
        for img_path in images:
            all_images.append(str(img_path))
            all_labels.append(class_idx)
    
    # Shuffle and split
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    all_images, all_labels = zip(*combined) if combined else ([], [])
    
    n_total = len(all_images)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    train_images = all_images[:n_train]
    train_labels = all_labels[:n_train]
    val_images = all_images[n_train:n_train + n_val]
    val_labels = all_labels[n_train:n_train + n_val]
    test_images = all_images[n_train + n_val:]
    test_labels = all_labels[n_train + n_val:]
    
    print(f"[Training] Dataset split: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")
    
    # Create datasets
    if use_albumentations:
        train_transform = get_train_augmentations(augment=True)
        val_transform = get_val_augmentations()
        train_dataset = AlbumentationsDataset(train_images, train_labels, train_transform)
        val_dataset = AlbumentationsDataset(val_images, val_labels, val_transform)
        if test_images:
            test_dataset = AlbumentationsDataset(test_images, test_labels, val_transform)
        else:
            test_dataset = val_dataset
    else:
        basic_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        train_dataset = PlantDiseaseDataset(train_images, train_labels, augment_transform)
        val_dataset = PlantDiseaseDataset(val_images, val_labels, basic_transform)
        if test_images:
            test_dataset = PlantDiseaseDataset(test_images, test_labels, basic_transform)
        else:
            test_dataset = val_dataset
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    # Create model
    print("\n[Training] Creating MobileNetV2 model...")
    model = PlantDiseaseMobileNetV2(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=freeze_backbone
    )
    model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Training] Total parameters: {total_params:,}")
    print(f"[Training] Trainable parameters: {trainable_params:,}")
    
    # Loss function with class weights
    class_counts = [train_labels.count(i) for i in range(num_classes)]
    class_weights = torch.tensor([1.0 / c if c > 0 else 1.0 for c in class_counts])
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Training loop
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': [],
        'best_epoch': 0,
        'best_val_acc': 0.0
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = output_dir / "best_model.pth"
    
    print("\n[Training] Starting training loop...")
    print("=" * 70)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            if isinstance(images, list):
                images = torch.stack(images)
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {running_loss / (batch_idx + 1):.4f} "
                      f"Acc: {running_correct / total_samples * 100:.2f}%")
        
        train_loss = running_loss / len(train_loader)
        train_acc = running_correct / total_samples
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                if isinstance(images, list):
                    images = torch.stack(images)
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Update scheduler
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_epoch'] = epoch + 1
            history['best_val_acc'] = best_val_acc
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_mapping,
                'num_classes': num_classes,
                'val_acc': val_acc,
                'epoch': epoch + 1
            }, best_model_path)
            print(f"  [SAVE] Best model saved (val_acc: {val_acc * 100:.2f}%)")
        else:
            patience_counter += 1
        
        # Print epoch summary
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | "
              f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n[Training] Early stopping triggered after {epoch + 1} epochs")
            break
    
    print("=" * 70)
    print(f"[Training] Training complete!")
    print(f"[Training] Best validation accuracy: {best_val_acc * 100:.2f}% (Epoch {history['best_epoch']})")
    
    # Save final model
    final_model_path = output_dir / "plant_disease_model.pth"
    
    # Load best model
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[Training] Loaded best model from epoch {checkpoint['epoch']}")
    
    # Save final model with metadata
    metadata = {
        "model_name": "MobileNetV2",
        "num_classes": num_classes,
        "image_size": [224, 224],
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "best_val_accuracy": best_val_acc,
        "best_epoch": history['best_epoch'],
        "training_date": datetime.now().isoformat(),
        "framework": "PyTorch",
        "dataset": "Kaggle Plant Disease Dataset",
        "version": "1.0.0",
        "epochs_completed": epoch + 1,
        "class_mapping": class_mapping,
        "device": str(device),
        "batch_size": batch_size,
        "initial_lr": learning_rate,
        "scheduler": "CosineAnnealingLR"
    }
    
    save_model(model, sorted(class_mapping.values()), str(final_model_path), metadata)
    
    # Save training history
    history_path = output_dir.parent / "reports" / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save dataset report
    dataset_report = {
        "total_images": len(all_images),
        "train_count": len(train_images),
        "val_count": len(val_images),
        "test_count": len(test_images),
        "num_classes": num_classes,
        "class_mapping": class_mapping,
        "class_counts": class_counts,
        "image_size": [224, 224],
        "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "random_seed": seed
    }
    
    dataset_report_path = output_dir.parent / "reports" / "dataset_report.json"
    with open(dataset_report_path, 'w') as f:
        json.dump(dataset_report, f, indent=2)
    
    print(f"\n[Training] Model saved to: {final_model_path}")
    print(f"[Training] Training history saved to: {history_path}")
    print(f"[Training] Dataset report saved to: {dataset_report_path}")
    
    # Evaluate on test set
    from backend.ai.disease_detection.evaluate import evaluate_model
    test_results = evaluate_model(model, test_loader, class_mapping, device, output_dir)
    
    return {
        "model_path": str(final_model_path),
        "history": history,
        "class_mapping": class_mapping,
        "num_classes": num_classes,
        "best_val_accuracy": best_val_acc,
        "test_results": test_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Plant Disease Detection Model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to extracted dataset")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    results = train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {results['model_path']}")
    print(f"Best validation accuracy: {results['best_val_accuracy'] * 100:.2f}%")
    print(f"Number of classes: {results['num_classes']}")