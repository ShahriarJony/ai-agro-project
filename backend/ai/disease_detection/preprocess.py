"""
Image Preprocessing Module for Plant Disease Detection
Implements OpenCV/PyTorch preprocessing pipeline with data augmentation.
"""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path
from typing import Tuple, Optional, List, Union
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet normalization constants (for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default target size
DEFAULT_IMAGE_SIZE = (224, 224)


def load_image_opencv(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """Load image using OpenCV and convert to RGB."""
    image_path = str(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    # OpenCV loads as BGR, convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_image_pil(image_path: Union[str, Path]) -> Optional[Image.Image]:
    """Load image using PIL."""
    try:
        img = Image.open(image_path).convert('RGB')
        return img
    except Exception:
        return None


def resize_image(image: np.ndarray, size: Tuple[int, int] = DEFAULT_IMAGE_SIZE, 
                 interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize image to target size."""
    return cv2.resize(image, size, interpolation=interpolation)


def normalize_image(image: np.ndarray, 
                    mean: List[float] = IMAGENET_MEAN, 
                    std: List[float] = IMAGENET_STD) -> np.ndarray:
    """Normalize image to [0, 1] then apply ImageNet normalization."""
    # Convert to float32 and scale to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    
    return image


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert numpy image (H, W, C) to PyTorch tensor (C, H, W)."""
    # Transpose from HWC to CHW
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
    return tensor


def preprocess_image_opencv(image_path: Union[str, Path], 
                             size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
                             normalize: bool = True) -> Optional[torch.Tensor]:
    """
    Complete preprocessing pipeline using OpenCV.
    
    Pipeline:
    1. Load image (BGR -> RGB)
    2. Resize to 224x224
    3. Normalize to [0, 1] + ImageNet stats
    4. Convert to tensor (C, H, W)
    """
    img = load_image_opencv(image_path)
    if img is None:
        return None
    
    img = resize_image(img, size)
    
    if normalize:
        img = normalize_image(img)
    
    tensor = image_to_tensor(img)
    return tensor


def preprocess_image_pil(image_path: Union[str, Path], 
                          size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
                          normalize: bool = True) -> Optional[torch.Tensor]:
    """
    Complete preprocessing pipeline using PIL/torchvision.
    """
    try:
        transform_list = [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
        if normalize:
            transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        
        transform = transforms.Compose(transform_list)
        img = load_image_pil(image_path)
        if img is None:
            return None
        
        tensor = transform(img)
        return tensor
        
    except Exception:
        return None


# Albumentations-based augmentation pipeline for training
def get_train_augmentations(size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
                             mean: List[float] = IMAGENET_MEAN,
                             std: List[float] = IMAGENET_STD,
                             augment: bool = True) -> A.Compose:
    """Get training augmentation pipeline using Albumentations."""
    
    if augment:
        transform_list = [
            A.Resize(size[0], size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.MotionBlur(blur_limit=3, p=0.1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    else:
        transform_list = [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    
    return A.Compose(transform_list)


def get_val_augmentations(size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
                           mean: List[float] = IMAGENET_MEAN,
                           std: List[float] = IMAGENET_STD) -> A.Compose:
    """Get validation/test augmentation pipeline (deterministic)."""
    return A.Compose([
        A.Resize(size[0], size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def apply_augmentation(image: np.ndarray, 
                       transform: A.Compose) -> Optional[torch.Tensor]:
    """Apply albumentations transform to numpy image."""
    try:
        augmented = transform(image=image)
        return augmented['image']
    except Exception:
        return None


class PreprocessingPipeline:
    """Unified preprocessing pipeline for training and inference."""
    
    def __init__(self, 
                 image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
                 mean: List[float] = IMAGENET_MEAN,
                 std: List[float] = IMAGENET_STD,
                 use_albumentations: bool = True):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.use_albumentations = use_albumentations
        
        if use_albumentations:
            self.train_transform = get_train_augmentations(image_size, mean, std, augment=True)
            self.val_transform = get_val_augmentations(image_size, mean, std)
        else:
            self.train_transform = None
            self.val_transform = None
    
    def preprocess_for_inference(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> Optional[torch.Tensor]:
        """Preprocess a single image for inference."""
        # Handle different input types
        if isinstance(image_input, (str, Path)):
            img = load_image_opencv(image_input)
        elif isinstance(image_input, np.ndarray):
            img = image_input
            if img.shape[2] == 3 and img.dtype == np.uint8:
                # Assume BGR if from OpenCV
                pass
            else:
                # Assume RGB
                pass
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input.convert('RGB'))
        else:
            return None
        
        if img is None:
            return None
        
        if self.use_albumentations and self.val_transform:
            tensor = apply_augmentation(img, self.val_transform)
        else:
            # Fallback to basic preprocessing
            img = resize_image(img, self.image_size)
            img = normalize_image(img, self.mean, self.std)
            tensor = image_to_tensor(img)
        
        # Add batch dimension
        if tensor is not None:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def preprocess_batch(self, image_paths: List[Union[str, Path]]) -> torch.Tensor:
        """Preprocess a batch of images."""
        tensors = []
        for path in image_paths:
            tensor = self.preprocess_for_inference(path)
            if tensor is not None:
                tensors.append(tensor)
        
        if tensors:
            return torch.cat(tensors, dim=0)
        else:
            return torch.empty(0, 3, *self.image_size)


def denormalize_image(tensor: torch.Tensor, 
                       mean: List[float] = IMAGENET_MEAN, 
                       std: List[float] = IMAGENET_STD) -> np.ndarray:
    """Denormalize tensor for visualization."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean
    
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy HWC
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    
    return image


def validate_preprocessing():
    """Test preprocessing pipeline."""
    print("Testing preprocessing pipeline...")
    
    # Create a test image
    test_img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    
    # Test basic preprocessing
    resized = resize_image(test_img, DEFAULT_IMAGE_SIZE)
    assert resized.shape == (224, 224, 3), f"Expected (224, 224, 3), got {resized.shape}"
    
    normalized = normalize_image(resized)
    assert normalized.dtype == np.float32, f"Expected float32, got {normalized.dtype}"
    assert -3 < normalized.mean() < 3, f"Normalized values out of expected range"
    
    tensor = image_to_tensor(normalized)
    assert tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor.shape}"
    
    # Test inference preprocessing
    pipeline = PreprocessingPipeline()
    tensor = pipeline.preprocess_for_inference(test_img)
    assert tensor is not None, "Preprocessing failed"
    assert tensor.shape == (1, 3, 224, 224), f"Expected (1, 3, 224, 224), got {tensor.shape}"
    
    print("All preprocessing tests passed!")


if __name__ == "__main__":
    validate_preprocessing()