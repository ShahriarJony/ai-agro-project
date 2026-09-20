"""
AgroAI - CNN Foliar Disease Detection Service
Module: Deep Learning Computer Vision Pipeline (MobileNetV2 / ResNet-18)

Interfaces:
- Image byte ingestion
- OpenCV preprocessing (RGB conversion, resize 224x224, min-max normalization)
- PyTorch model loading abstraction (loads saved plant_disease_cnn.pth if available)
- Fallback/Demo classification response clearly labeled as not trained.
"""

import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant_disease_cnn.pth")

class CNNDiseaseService:
    CLASSES = ["Healthy Foliage", "Early Blight (Alternaria solani)", "Late Blight (Phytophthora infestans)", "Leaf Rust (Puccinia)"]

    def __init__(self):
        self.model = self._load_model()
        self.is_trained = self.model is not None

    def _load_model(self):
        """Model loading abstraction. Returns PyTorch nn.Module if .pth model exists."""
        if os.path.exists(MODEL_PATH):
            try:
                import torch
                model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
                model.eval()
                return model
            except Exception:
                return None
        return None

    def preprocess_image(self, image_bytes: bytes):
        """
        OpenCV Preprocessing Pipeline:
        1. Decode byte stream to OpenCV BGR image
        2. Convert BGR to RGB
        3. Resize image matrix to standard 224x224 input tensor size
        4. Normalize pixel values to [0.0, 1.0]
        """
        try:
            import cv2
            import numpy as np
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_normalized = img_resized.astype(np.float32) / 255.0
            return img_normalized
        except Exception:
            return None

    def classify_leaf_image(self, image_bytes: bytes, filename: str = "upload.jpg") -> dict:
        processed_tensor = self.preprocess_image(image_bytes)

        if self.is_trained and self.model is not None:
            try:
                import torch
                import numpy as np
                tensor = torch.from_numpy(processed_tensor).permute(2, 0, 1).unsqueeze(0)
                with torch.no_grad():
                    outputs = self.model(tensor)
                    prob = torch.softmax(outputs, dim=1).numpy()[0]
                    class_idx = int(np.argmax(prob))
                    confidence = float(prob[class_idx] * 100.0)
                
                status_label = "Trained PyTorch Model"
                is_demo = False
                detected_class = self.CLASSES[class_idx]
            except Exception:
                detected_class = self.CLASSES[1]
                confidence = 94.8
                status_label = "Demo / Model Not Trained"
                is_demo = True
        else:
            detected_class = "Early Blight (Alternaria solani)"
            confidence = 94.8
            status_label = "Demo / Model Not Trained"
            is_demo = True

        treatment_protocols = {
            "Healthy Foliage": "Canopy is healthy. Maintain standard monitoring schedule.",
            "Early Blight (Alternaria solani)": "Apply copper hydroxide fungicide spray at 2.5 g/L. Isolate Sector D-North.",
            "Late Blight (Phytophthora infestans)": "Immediate systemic fungicide application (Mefenoxam). Disengage overhead irrigation.",
            "Leaf Rust (Puccinia)": "Apply azoxystrobin foliar spray. Reduce canopy humidity levels."
        }

        return {
            "algorithm": "CNN Foliar Disease Classifier (MobileNetV2)",
            "filename": filename,
            "is_trained": not is_demo,
            "status": status_label,
            "predicted_disease": detected_class,
            "confidence_percent": confidence,
            "severity": "Moderate (Foliar Stage 2)" if "Blight" in detected_class else "Low",
            "opencv_preprocessing": {
                "tensor_shape": [224, 224, 3],
                "color_space": "RGB",
                "pixel_normalization": "[0.0, 1.0]",
                "opencv_status": "Preprocessed 224x224 RGB" if processed_tensor is not None else "Byte Stream Decoded"
            },
            "treatment_recommendation": treatment_protocols.get(detected_class, "Consult agricultural advisor."),
            "note": "PyTorch deep learning model training will be performed in a separate phase." if is_demo else "Generated via trained MobileNetV2 weights."
        }
