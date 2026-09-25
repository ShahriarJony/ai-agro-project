"""
CNN Inference Service for Plant Disease Detection
Provides model loading and prediction interface for FastAPI backend.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.ai.disease_detection.model import load_model
from backend.ai.disease_detection.preprocess import (
    PreprocessingPipeline, preprocess_image_opencv, 
    IMAGENET_MEAN, IMAGENET_STD
)


class CNNDiseaseDetector:
    """Inference service for plant disease detection using trained MobileNetV2."""
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 class_names_path: Optional[str] = None,
                 metadata_path: Optional[str] = None,
                 image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the disease detector.
        
        Args:
            model_path: Path to trained model .pth file
            class_names_path: Path to class names JSON
            metadata_path: Path to model metadata JSON
            image_size: Target image size for preprocessing
        """
        self.model = None
        self.class_names: List[str] = []
        self.class_mapping: Dict[str, int] = {}
        self.metadata: Dict[str, Any] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.preprocessor = PreprocessingPipeline(image_size=image_size)
        self.is_loaded = False
        
        # Set default paths if not provided
        if model_path is None:
            model_path = Path(__file__).parent / "models" / "plant_disease_model.pth"
        if class_names_path is None:
            class_names_path = Path(__file__).parent / "models" / "class_names.json"
        if metadata_path is None:
            metadata_path = Path(__file__).parent / "models" / "model_metadata.json"
        
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self.metadata_path = Path(metadata_path)
        
        # Try to load model on initialization
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load the trained model and associated files."""
        if not self.model_path.exists():
            print(f"[Inference] Model not found at: {self.model_path}")
            print(f"[Inference] Model will not be loaded. Inference will return demo results.")
            return False
        
        try:
            # Load class names
            if self.class_names_path.exists():
                with open(self.class_names_path, 'r') as f:
                    mapping = json.load(f)
                    self.class_names = [mapping[str(i)] for i in range(len(mapping))]
                    self.class_mapping = {name: i for i, name in enumerate(self.class_names)}
            
            # Load metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            # Load model
            loaded = load_model(str(self.model_path), self.device)
            self.model = loaded['model']
            self.class_names = loaded.get('class_names', self.class_names)
            self.metadata = loaded.get('metadata', self.metadata)
            
            self.is_loaded = True
            print(f"[Inference] Model loaded successfully")
            print(f"[Inference] Classes: {len(self.class_names)}")
            print(f"[Inference] Device: {self.device}")
            return True
            
        except Exception as e:
            print(f"[Inference] Failed to load model: {e}")
            self.is_loaded = False
            return False
    
    def preprocess_image(self, image_bytes: bytes) -> Optional[torch.Tensor]:
        """Preprocess image bytes for inference."""
        import cv2
        
        try:
            # Decode bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, self.image_size)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
            std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
            img = (img - mean) / std
            
            # Convert to tensor (C, H, W)
            tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            print(f"[Inference] Preprocessing failed: {e}")
            return None
    
    def predict(self, 
                image_bytes: bytes,
                filename: str = "upload.jpg",
                top_k: int = 3,
                threshold: float = 0.5) -> Dict[str, Any]:
        """
        Run disease prediction on an image.
        
        Args:
            image_bytes: Raw image bytes
            filename: Original filename
            top_k: Number of top predictions to return
            threshold: Confidence threshold
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Preprocess
        tensor = self.preprocess_image(image_bytes)
        if tensor is None:
            return {
                "success": False,
                "error": "Failed to preprocess image",
                "filename": filename,
                "is_trained": False
            }
        
        # Check if model is loaded
        if not self.is_loaded or self.model is None:
            return {
                "success": False,
                "error": "Model not loaded",
                "filename": filename,
                "is_trained": False
            }
        
        try:
            # Move tensor to device
            tensor = tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, top_k)
                top_probs = top_probs[0].cpu().numpy()
                top_indices = top_indices[0].cpu().numpy()
            
            # Get class names
            predictions = []
            for i in range(top_k):
                class_idx = int(top_indices[i])
                class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Unknown_{class_idx}"
                confidence = float(top_probs[i] * 100.0)
                
                # Normalize class name for display
                display_name = normalize_class_name(class_name)
                
                predictions.append({
                    "rank": i + 1,
                    "class_index": class_idx,
                    "class_name": class_name,
                    "display_name": display_name,
                    "confidence": confidence / 100.0,
                    "confidence_percent": confidence
                })
            
            # Get best prediction
            best_pred = predictions[0]
            is_healthy = "healthy" in best_pred['class_name'].lower()
            
            # Get disease info
            disease_info = get_disease_info(best_pred['class_name'])
            
            inference_time = time.time() - start_time
            
            return {
                "success": True,
                "is_trained": True,
                "model": self.metadata.get("model_name", "MobileNetV2"),
                "model_version": self.metadata.get("version", "1.0.0"),
                "filename": filename,
                "predicted_disease": best_pred['class_name'],
                "display_name": best_pred['display_name'],
                "confidence": best_pred['confidence'],
                "confidence_percent": best_pred['confidence_percent'],
                "is_healthy": is_healthy,
                "disease_info": disease_info,
                "top_predictions": predictions,
                "inference_time_ms": round(inference_time * 1000, 2),
                "image_size": list(self.image_size),
                "cv_preprocessing": {
                    "tensor_shape": [1, 3, self.image_size[0], self.image_size[1]],
                    "color_space": "RGB",
                    "normalization": "ImageNet stats"
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Inference failed: {str(e)}",
                "filename": filename,
                "is_trained": self.is_loaded
            }
    
    def predict_from_path(self, image_path: str) -> Dict[str, Any]:
        """Run prediction from image file path."""
        from backend.ai.disease_detection.preprocess import load_image_opencv
        
        img = load_image_opencv(image_path)
        if img is None:
            # Try reading bytes
            with open(image_path, 'rb') as f:
                img_bytes = f.read()
            return self.predict(img_bytes, filename=Path(image_path).name)
        
        # Convert numpy array to bytes
        import cv2
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return self.predict(buffer.tobytes(), filename=Path(image_path).name)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get model status information."""
        return {
            "model_loaded": self.is_loaded,
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "num_classes": len(self.class_names),
            "class_names": self.class_names if self.is_loaded else [],
            "device": str(self.device),
            "metadata": self.metadata if self.metadata else {},
            "image_size": list(self.image_size)
        }


def normalize_class_name(class_name: str) -> str:
    """Normalize class name for display."""
    # Handle common formats like "Tomato___Late_blight"
    if '___' in class_name:
        parts = class_name.replace('___', ' ').replace('_', ' ')
        parts = parts.split()
        if len(parts) > 1:
            crop = parts[0]
            disease = ' '.join(parts[1:])
            # Capitalize properly
            crop = crop.capitalize()
            disease = ' '.join(word.capitalize() for word in disease.split())
            return f"{crop} - {disease}"
        return parts.capitalize()
    
    # Already formatted
    if '_' in class_name:
        return class_name.replace('_', ' ').title()
    
    return class_name


# Disease Information Database
DISEASE_DATABASE: Dict[str, Dict[str, Any]] = {
    # Apple diseases
    "Apple___healthy": {
        "crop": "Apple",
        "scientific_name": "Malus domestica",
        "description": "Healthy apple leaves show vibrant green coloration with no visible lesions or spots.",
        "symptoms": "None - foliage appears normal and healthy.",
        "severity": "None",
        "treatment": "Maintain regular monitoring and proper orchard care.",
        "prevention": "Regular pruning, adequate fertilization, and pest control.",
        "source": "PlantVillage Database"
    },
    "Apple___Apple_scab": {
        "crop": "Apple",
        "scientific_name": "Malus domestica",
        "description": "Apple scab is a serious disease caused by the fungus Venturia inaequalis.",
        "symptoms": "Olive-green to brownish-black circular spots on leaves, thickened appearance.",
        "severity": "Moderate to Severe",
        "treatment": "Apply fungicides such as sulfur or synthetic fungicides during bloom and throughout the growing season.",
        "prevention": "Use resistant varieties, prune for air circulation, and apply fungicide preventatively.",
        "pesticide": "Sulfur-based fungicides or myclobutanil",
        "source": "PlantVillage Database"
    },
    # Tomato diseases
    "Tomato___healthy": {
        "crop": "Tomato",
        "scientific_name": "Solanum lycopersicum",
        "description": "Healthy tomato leaves are bright green with no visible spots or discoloration.",
        "symptoms": "None.",
        "severity": "None",
        "treatment": "Continue regular care and monitoring.",
        "prevention": "Maintain proper watering, fertilization, and pest control.",
        "source": "PlantVillage Database"
    },
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "scientific_name": "Solanum lycopersicum",
        "description": "Late blight is a devastating disease caused by Phytophthora infestans.",
        "symptoms": "Irregular brown or black spots on leaves, white fuzzy growth on underside, rapidly spreading lesions.",
        "severity": "Severe",
        "treatment": "Immediately remove infected plant parts. Apply fungicides containing mefenoxim or chlorothalonil.",
        "prevention": "Avoid overhead irrigation, ensure good air circulation, use resistant varieties, and apply preventative fungicides.",
        "pesticide": "Mefenoxam or chlorothalonil at 1.8 L/ha",
        "source": "PlantVillage Database"
    },
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "scientific_name": "Solanum lycopersicum",
        "description": "Early blight is caused by the fungus Alternaria solani and is one of the mostcommon tomato diseases.",
        "symptoms": "Brown spots with concentric rings (target-like appearance) on lower leaves first, yellowing of leaves.",
        "severity": "Moderate to Severe",
        "treatment": "Remove infected leaves, apply copper-based fungicides or chlorothalonil.",
        "prevention": "Crop rotation, resistant varieties, mulching, and avoiding leaf wetness.",
        "pesticide": "Copper hydroxide at 2.5 g/L",
        "source": "PlantVillage Database"
    },
    "Tomato___Leaf_Mold": {
        "crop": "Tomato",
        "scientific_name": "Solanum lycopersicum",
        "description": "Leaf mold is caused by the fungus Passalora fulva (formerly Cladosporium fulvum).",
        "symptoms": "Yellowish areas on upper surface of leaves, fuzzy brown mold on underside.",
        "severity": "Moderate",
        "treatment": "Apply fungicides, remove heavily infected leaves.",
        "prevention": "Increase air circulation, avoid overhead watering, use resistant varieties.",
        "pesticide": "Benomyl or chlorothalonil",
        "source": "PlantVillage Database"
    },
    "Tomato___Septoria_leaf_spot": {
        "crop": "Tomato",
        "scientific_name": "Solanum lycopersicum",
        "description": "Septoria leaf spot is caused by the fungus Septoria lycopersici.",
        "symptoms": "Small circular spots with dark borders and light centers on lower leaves, yellowing.",
        "severity": "Moderate",
        "treatment": "Remove infected leaves, apply appropriate fungicides.",
        "prevention": "Crop rotation, resistant varieties, proper spacing for air circulation.",
        "pesticide": "Copper-based fungicides",
        "source": "PlantVillage Database"
    },
    "Tomato___Tomato_mosaic_virus": {
        "crop": "Tomato",
        "scientific_name": "Solanum lycopersicum",
        "description": "Tomato mosaic virus causes mottling and distortion of leaves.",
        "symptoms": "Light green to yellow mottling on leaves, leaf distortion, stunted growth.",
        "severity": "Moderate",
        "treatment": "No chemical cure. Remove infected plants.",
        "prevention": "Use virus-free seed, control insect vectors, sanitation.",
        "pesticide": "None - remove infected plants",
        "source": "PlantVillage Database"
    },
    "Tomato___Tomato_Yellow Leaf Curl Virus": {
        "crop": "Tomato",
        "scientific_name": "Solanum lycopersicum",
        "description": "Tomato yellow leaf curl virus is transmitted by whiteflies.",
        "symptoms": "Yellowing and curling of leaves, stunted growth, reduced fruit production.",
        "severity": "Severe",
        "treatment": "No cure. Remove infected plants and control whitefly vectors.",
        "prevention": "Use resistant varieties, control whiteflies with insecticides, use reflective mulches.",
        "pesticide": "Imidacloprid for whitefly control",
        "source": "PlantVillage Database"
    },
    # Potato diseases
    "Potato___healthy": {
        "crop": "Potato",
        "scientific_name": "Solanum tuberosum",
        "description": "Healthy potato foliage is dark green and vigorous.",
        "symptoms": "None.",
        "severity": "None",
        "treatment": "Regular monitoring and care.",
        "prevention": "Standard potato cultivation practices.",
        "source": "PlantVillage Database"
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "scientific_name": "Solanum tuberosum",
        "description": "Potato late blight is caused by Phytophthora infestans.",
        "symptoms": "Water-soaked spots on leaves that turn brown, white mold on underside in humid weather.",
        "severity": "Severe",
        "treatment": "Apply fungicides immediately. Remove infected tubers.",
        "prevention": "Resistant varieties, proper spacing, fungicide applications during wet periods.",
        "pesticide": "Mefenoxam or metalaxyl-based fungicides",
        "source": "PlantVillage Database"
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "scientific_name": "Solanum tuberosum",
        "description": "Potato early blight is caused by Alternaria solani.",
        "symptoms": "Brown spots with concentric rings, yellowing of lower leaves.",
        "severity": "Moderate to Severe",
        "treatment": "Apply fungicides, remove infected plant debris.",
        "prevention": "Crop rotation, resistant varieties, mulching.",
        "pesticide": "Chlorothalonil or copper-based fungicides",
        "source": "PlantVillage Database"
    },
    # Other common classes
    "healthy": {
        "crop": "Multiple",
        "scientific_name": "Various",
        "description": "Healthy plant tissue with no visible disease symptoms.",
        "symptoms": "None - foliage appears normal.",
        "severity": "None",
        "treatment": "Maintain standard care and monitoring.",
        "prevention": "Regular inspection and proper growing conditions.",
        "source": "PlantVillage Database"
    }
}


def get_disease_info(class_name: str) -> Dict[str, Any]:
    """Get disease information for a predicted class."""
    # Try exact match
    if class_name in DISEASE_DATABASE:
        info = DISEASE_DATABASE[class_name].copy()
        info["source_class"] = class_name
        return info
    
    # Try variations
    variations = [
        class_name.replace(' ', '_'),
        class_name.replace(' ', '_').lower(),
        class_name.lower(),
        class_name.replace('___', '_')
    ]
    
    for variant in variations:
        if variant in DISEASE_DATABASE:
            info = DISEASE_DATABASE[variant].copy()
            info["source_class"] = class_name
            return info
    
    # Return unknown disease info
    return {
        "crop": "Unknown",
        "scientific_name": "Unknown",
        "description": f"No specific information available for this disease: {class_name}",
        "symptoms": "Consult local agricultural extension service.",
        "severity": "Unknown",
        "treatment": "Please consult with an agricultural advisor for proper diagnosis.",
        "prevention": "General plant care practices apply.",
        "source": "No specific data available",
        "source_class": class_name
    }


def get_recommendation(disease_class: str, severity: str = "Moderate") -> Dict[str, Any]:
    """Get treatment recommendation for a disease."""
    info = get_disease_info(disease_class)
    
    recommendation = {
        "disease_name": disease_class,
        "severity": info.get("severity", severity),
        "treatment": info.get("treatment", "Consult agricultural advisor."),
        "prevention": info.get("prevention", "General plant care practices."),
        "pesticide": info.get("pesticide", None),
        "safety_notes": info.get("safety_notes", "Follow all pesticide label instructions."),
        "requires_professional_consultation": info.get("crop", "") == "Unknown"
    }
    
    return recommendation


# Global detector instance for FastAPI
_detector_instance = None

def get_detector(model_path: Optional[str] = None) -> CNNDiseaseDetector:
    """Get singleton disease detector instance."""
    global _detector_instance
    if _detector_instance is None or model_path:
        _detector_instance = CNNDiseaseDetector(model_path=model_path)
    return _detector_instance


def reset_detector():
    """Reset the global detector instance."""
    global _detector_instance
    _detector_instance = None


if __name__ == "__main__":
    # Test inference service
    print("Testing CNN Inference Service...")
    
    detector = CNNDiseaseDetector()
    status = detector.get_model_status()
    print(f"Model status: {status}")
    
    if status['model_loaded']:
        print("Model is loaded and ready")
    else:
        print("Model not loaded - will return demo results")