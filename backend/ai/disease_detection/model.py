"""
Model Architecture Module for Plant Disease Detection
Implements MobileNetV2 with transfer learning for disease classification.
"""

import os
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, Any, List
from pathlib import Path
import json


class PlantDiseaseMobileNetV2(nn.Module):
    """MobileNetV2 adapted for plant disease classification."""
    
    def __init__(self, 
                 num_classes: int,
                 pretrained: bool = True,
                 dropout_rate: float = 0.2,
                 freeze_backbone: bool = False):
        """
        Initialize MobileNetV2 for plant disease detection.
        
        Args:
            num_classes: Number of disease classes
            pretrained: Use ImageNet pretrained weights
            dropout_rate: Dropout rate for classifier
            freeze_backbone: Freeze backbone weights (only train classifier)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        
        # Load MobileNetV2 backbone
        if pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            self.backbone = models.mobilenet_v2(weights=weights)
        else:
            self.backbone = models.mobilenet_v2(weights=None)
        
        # Get the feature extractor (everything before classifier)
        self.features = self.backbone.features
        
        # Get the number of features from the last layer
        # MobileNetV2 last channel is 1280
        last_channel = self.backbone.last_channel
        
        # Replace classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(last_channel, num_classes)
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
        
        # Initialize new classifier weights
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.features.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        # Global average pooling
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x


class PlantDiseaseResNet(nn.Module):
    """ResNet alternative for plant disease classification."""
    
    def __init__(self,
                 num_classes: int,
                 pretrained: bool = True,
                 architecture: str = 'resnet18',
                 freeze_backbone: bool = False):
        super().__init__()
        
        self.num_classes = num_classes
        
        if architecture == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        elif architecture == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
        elif architecture == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Replace final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
        
        if freeze_backbone:
            # Freeze all except final layer
            for name, param in self.backbone.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def create_model(model_name: str,
                 num_classes: int,
                 pretrained: bool = True,
                 **kwargs) -> nn.Module:
    """
    Factory function to create model.
    
    Args:
        model_name: 'mobilenetv2', 'resnet18', 'resnet34', 'resnet50'
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        **kwargs: Additional arguments
    
    Returns:
        PyTorch model
    """
    model_name = model_name.lower()
    
    if model_name == 'mobilenetv2':
        return PlantDiseaseMobileNetV2(num_classes, pretrained, **kwargs)
    elif model_name in ('resnet18', 'resnet34', 'resnet50'):
        return PlantDiseaseResNet(num_classes, pretrained, model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: mobilenetv2, resnet18, resnet34, resnet50")


def save_model(model: nn.Module,
               class_names: List[str],
               model_path: str,
               metadata: Optional[Dict[str, Any]] = None):
    """
    Save model with class names and metadata.
    
    Args:
        model: PyTorch model
        class_names: List of class names in order
        model_path: Path to save model (.pth)
        metadata: Additional metadata to save
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': len(class_names),
        'model_architecture': model.__class__.__name__,
        'metadata': metadata or {}
    }, model_path)
    
    # Save class names separately for easy access
    class_names_path = model_path.parent / "class_names.json"
    with open(class_names_path, 'w') as f:
        json.dump({str(i): name for i, name in enumerate(class_names)}, f, indent=2)
    
    # Save metadata
    if metadata:
        metadata_path = model_path.parent / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"[Model] Saved model to: {model_path}")
    print(f"[Model] Saved class names to: {class_names_path}")


def load_model(model_path: str,
               device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load model from checkpoint.
    
    Returns:
        Dict with 'model', 'class_names', 'metadata', 'num_classes'
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    
    class_names = checkpoint.get('class_names', [])
    num_classes = checkpoint.get('num_classes', len(class_names))
    model_arch = checkpoint.get('model_architecture', 'PlantDiseaseMobileNetV2')
    metadata = checkpoint.get('metadata', {})
    
    # Create model
    if model_arch == 'PlantDiseaseMobileNetV2':
        model = PlantDiseaseMobileNetV2(num_classes, pretrained=False)
    elif model_arch.startswith('PlantDiseaseResNet'):
        arch_name = model_arch.replace('PlantDiseaseResNet', 'resnet').lower()
        if arch_name not in ('resnet18', 'resnet34', 'resnet50'):
            arch_name = 'resnet18'
        model = PlantDiseaseResNet(num_classes, pretrained=False, architecture=arch_name)
    else:
        model = PlantDiseaseMobileNetV2(num_classes, pretrained=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"[Model] Loaded model from: {model_path}")
    print(f"[Model] Classes: {num_classes}, Device: {device}")
    
    return {
        'model': model,
        'class_names': class_names,
        'metadata': metadata,
        'num_classes': num_classes,
        'device': device
    }


def get_model_summary(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> str:
    """Get model summary string."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Try to run a forward pass to verify
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        output_shape = tuple(output.shape)
        forward_ok = True
    except Exception as e:
        output_shape = str(e)
        forward_ok = False
    
    summary = f"""
Model Summary:
==============
Architecture: {model.__class__.__name__}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Non-trainable Parameters: {total_params - trainable_params:,}
Input Size: {input_size}
Output Shape: {output_shape}
Forward Pass: {'OK' if forward_ok else 'FAILED'}
Device: {device}
"""
    return summary


def count_parameters_by_layer(model: nn.Module) -> Dict[str, int]:
    """Count parameters per layer."""
    params = {}
    for name, param in model.named_parameters():
        layer_name = '.'.join(name.split('.')[:-1]) if '.' in name else name
        if layer_name not in params:
            params[layer_name] = 0
        params[layer_name] += param.numel()
    return params


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Test MobileNetV2
    model = create_model('mobilenetv2', num_classes=10, pretrained=False)
    print(f"MobileNetV2 created: {model}")
    print(get_model_summary(model))
    
    # Test ResNet
    model = create_model('resnet18', num_classes=10, pretrained=False)
    print(f"ResNet18 created: {model}")
    print(get_model_summary(model))
    
    # Test save/load
    class_names = [f"Class_{i}" for i in range(10)]
    save_model(model, class_names, "test_model.pth", {"test": True})
    
    loaded = load_model("test_model.pth")
    print(f"Loaded model with {len(loaded['class_names'])} classes")
    
    # Cleanup
    os.remove("test_model.pth")
    os.remove("class_names.json")
    os.remove("model_metadata.json")
    
    print("All model tests passed!")