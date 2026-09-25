"""
Plant Disease Detection Module
Backend AI module for plant leaf disease classification using MobileNetV2.
"""

from backend.ai.disease_detection.kaggle_dataset import KaggleDatasetManager, KaggleDataset
from backend.ai.disease_detection.dataset import PlantDiseaseDataset, ImageRecord, DatasetManifest
from backend.ai.disease_detection.preprocess import (
    PreprocessingPipeline, preprocess_image_opencv, preprocess_image_pil,
    resize_image, normalize_image, image_to_tensor,
    get_train_augmentations, get_val_augmentations,
    IMAGENET_MEAN, IMAGENET_STD, DEFAULT_IMAGE_SIZE
)
from backend.ai.disease_detection.model import (
    PlantDiseaseMobileNetV2, PlantDiseaseResNet,
    create_model, save_model, load_model, get_model_summary
)
from backend.ai.disease_detection.train import train_model, set_seed, get_device
from backend.ai.disease_detection.evaluate import (
    evaluate_model, analyze_class_imbalance,
    print_detailed_classification_report
)
from backend.ai.disease_detection.inference import (
    CNNDiseaseDetector, normalize_class_name, get_disease_info, get_recommendation,
    get_detector, reset_detector
)

__all__ = [
    'KaggleDatasetManager',
    'KaggleDataset',
    'PlantDiseaseDataset',
    'ImageRecord',
    'DatasetManifest',
    'PreprocessingPipeline',
    'PlantDiseaseMobileNetV2',
    'PlantDiseaseResNet',
    'create_model',
    'save_model',
    'load_model',
    'get_model_summary',
    'train_model',
    'set_seed',
    'get_device',
    'evaluate_model',
    'analyze_class_imbalance',
    'print_detailed_classification_report',
    'CNNDiseaseDetector',
    'normalize_class_name',
    'get_disease_info',
    'get_recommendation',
    'get_detector',
    'reset_detector',
]

__version__ = "1.0.0"