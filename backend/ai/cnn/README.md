# AgroAI — CNN Leaf Disease Detection Research Foundation

## Overview
This module outlines the computer vision deep learning pipeline for plant foliage disease identification.

## Pipeline Architecture
```text
Leaf Image (RGB)
      ↓
Resize (224x224)
      ↓
Normalize (Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225])
      ↓
Tensor Conversion
      ↓
CNN Model (MobileNetV2 / ResNet18)
      ↓
Disease Class Probability (Softmax)
```

## Recommended Academic Model
For our university laboratory project, we recommend **MobileNetV2** transfer learning:
- **Lightweight**: Fast inference on CPU/mobile devices.
- **Pre-trained**: Uses ImageNet weights for rapid convergence.
- **Accuracy**: Highly effective for PlantVillage leaf image datasets.
