"""
Configuration Module for Plant Disease Detection
Centralizes configuration for dataset, training, and inference.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    # Model parameters
    model_name: str = "MobileNetV2"
    num_classes: int = 0
    image_size: int = 224
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.2
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    early_stop_patience: int = 5
    random_seed: int = 42
    
    # Data parameters
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Augmentation
    use_albumentations: bool = True
    augmentation_enabled: bool = True
    
    # Optimizer
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"
    min_lr: float = 1e-6
    
    # Paths
    model_output_dir: str = "models"
    report_output_dir: str = "reports"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class InferenceConfig:
    """Inference configuration parameters."""
    model_path: str = "backend/ai/disease_detection/models/plant_disease_model.pth"
    class_names_path: str = "backend/ai/disease_detection/models/class_names.json"
    metadata_path: str = "backend/ai/disease_detection/models/model_metadata.json"
    image_size: int = 224
    confidence_threshold: float = 0.5
    top_k: int = 3
    device: str = "auto"  # auto, cpu, or cuda
    enable_severity: bool = True


@dataclass
class KaggleConfig:
    """Kaggle API configuration."""
    username: Optional[str] = None
    api_key: Optional[str] = None
    dataset_ref: Optional[str] = None
    cache_dir: str = "backend/ai/disease_detection/data/cache"
    
    def is_configured(self) -> bool:
        return self.username is not None and self.api_key is not None


class Config:
    """Main configuration manager for plant disease detection."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.module_dir = Path(__file__).parent
        
        # Load from environment
        self.kaggle = KaggleConfig(
            username=os.environ.get("KAGGLE_USERNAME"),
            api_key=os.environ.get("KAGGLE_KEY"),
            dataset_ref=os.environ.get("DISEASE_DATASET"),
            cache_dir=os.environ.get("DATASET_CACHE_DIR", str(self.module_dir / "data" / "cache"))
        )
        
        self.training = TrainingConfig(
            model_output_dir=str(self.module_dir / "models"),
            report_output_dir=str(self.module_dir / "reports")
        )
        
        self.inference = InferenceConfig(
            model_path=str(self.module_dir / "models" / "plant_disease_model.pth"),
            class_names_path=str(self.module_dir / "models" / "class_names.json"),
            metadata_path=str(self.module_dir / "models" / "model_metadata.json")
        )
    
    def save_training_config(self) -> str:
        """Save training configuration to file."""
        config_path = Path(self.training.model_output_dir) / "training_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            "training": self.training.to_dict(),
            "inference": asdict(self.inference),
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        return str(config_path)
    
    def load_training_config(self, path: str) -> TrainingConfig:
        """Load training configuration from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        if "training" in data:
            return TrainingConfig.from_dict(data["training"])
        return TrainingConfig.from_dict(data)
    
    def get_device(self):
        """Get the best available torch device."""
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        return device
    
    def get_class_mapping_path(self) -> str:
        """Get the path where class mapping should be stored."""
        return str(self.module_dir / "models" / "class_names.json")
    
    def get_model_path(self) -> str:
        """Get the path where the trained model should be stored."""
        return str(self.module_dir / "models" / "plant_disease_model.pth")
    
    def get_metadata_path(self) -> str:
        """Get the path where model metadata should be stored."""
        return str(self.module_dir / "models" / "model_metadata.json")


# Default singleton config
DEFAULT_CONFIG = Config()


def get_config() -> Config:
    """Get the default config singleton."""
    return DEFAULT_CONFIG


def validate_kaggle_auth() -> bool:
    """Validate Kaggle authentication is available."""
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    
    if not username or not key:
        print("[Config] KAGGLE_USERNAME or KAGGLE_KEY environment variables not set")
        return False
    
    return True


def get_env_config() -> Dict[str, Any]:
    """Get configuration from environment variables."""
    return {
        "kaggle_username": os.environ.get("KAGGLE_USERNAME"),
        "kaggle_key_set": os.environ.get("KAGGLE_KEY") is not None,
        "dataset": os.environ.get("DISEASE_DATASET"),
        "cache_dir": os.environ.get("DATASET_CACHE_DIR"),
        "model_path": os.environ.get("MODEL_PATH"),
        "class_names_path": os.environ.get("CLASS_NAMES_PATH"),
        "firebase_project_id": os.environ.get("FIREBASE_PROJECT_ID"),
        "firebase_api_key": "***" if os.environ.get("FIREBASE_API_KEY") else None,
    }


# DANGER_CLASSES - Classes that indicate severe disease requiring immediate action
SEVERE_DISEASE_CLASSES = [
    "Late_blight",
    "Tomato___Late_blight",
    "Potato___Late_blight",
    "Yellow_Leaf_Curl",
    "Tomato_Yellow Leaf Curl Virus",
]


def is_severe_disease(class_name: str) -> bool:
    """Check if a disease class indicates severe infection."""
    class_lower = class_name.lower().replace('___', '_').replace(' ', '_')
    for severe in SEVERE_DISEASE_CLASSES:
        severe_check = severe.lower().replace('___', '_').replace(' ', '_')
        if severe_check in class_lower:
            return True
    return False


def get_severity_level(class_name: str) -> str:
    """Determine severity level for a disease class."""
    if "healthy" in class_name.lower():
        return "None"
    if is_severe_disease(class_name):
        return "Severe"
    if "blight" in class_name.lower() or "mosaic" in class_name.lower():
        return "Moderate to Severe"
    if "rust" in class_name.lower() or "spot" in class_name.lower():
        return "Moderate"
    return "Low to Moderate"


if __name__ == "__main__":
    # Test configuration
    config = Config()
    print("Configuration loaded:")
    print(f"  Kaggle configured: {config.kaggle.is_configured()}")
    print(f"  Device: {config.get_device()}")
    print(f"  Model path: {config.get_model_path()}")
    print(f"  Training config saved to: {config.save_training_config()}")