"""
Model Evaluation Module for Plant Disease Detection
Evaluates trained model on test dataset with comprehensive metrics.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, top_k_accuracy_score
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model: nn.Module,
                   test_loader: torch.utils.data.DataLoader,
                   class_mapping: Dict[int, str],
                   device: torch.device,
                   output_dir: str = "reports") -> Dict[str, Any]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        class_mapping: Mapping of class IDs to names
        device: Torch device
        output_dir: Directory to save evaluation artifacts
    
    Returns:
        Dictionary with evaluation metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Evaluation] Running evaluation on test set...")
    
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_outputs = []
    inference_times = []
    
    import time
    
    with torch.no_grad():
        for images, labels in test_loader:
            if isinstance(images, list):
                images = torch.stack(images)
            images = images.to(device)
            labels = labels.to(device)
            
            start_time = time.time()
            outputs = model(images)
            inference_times.append(time.time() - start_time)
            
            all_outputs.append(outputs.cpu())
            all_probs.append(torch.softmax(outputs, dim=1).cpu())
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = torch.cat(all_probs).numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Per-class metrics (handle classes with zero samples)
    present_classes = np.unique(np.concatenate([all_labels, all_preds]))
    class_names = [class_mapping.get(i, f"Class_{i}") for i in sorted(present_classes)]
    
    precision_per_class = precision_score(all_labels, all_preds, 
                                          labels=list(present_classes), 
                                          average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds,
                                   labels=list(present_classes),
                                   average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds,
                            labels=list(present_classes),
                            average=None, zero_division=0)
    
    # Macro and weighted averages
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Top-3 accuracy
    top3_preds = np.argsort(all_probs, axis=1)[:, -3:]
    top3_correct = np.mean([all_labels[i] in top3_preds[i] for i in range(len(all_labels))])
    
    # Classification report
    report = classification_report(all_labels, all_preds, 
                                   labels=list(present_classes),
                                   target_names=class_names,
                                   output_dict=True,
                                   zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(present_classes))
    
    # Generate confusion matrix plot
    plot_path = output_dir / "confusion_matrix.png"
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Plant Disease Detection')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Evaluation] Confusion matrix saved to: {plot_path}")
    except Exception as e:
        print(f"[Evaluation] Could not save confusion matrix plot: {e}")
    
    # Per-class metrics
    per_class_metrics = {}
    for i, cls_name in enumerate(class_names):
        cls_id = sorted(present_classes)[i]
        per_class_metrics[cls_name] = {
            "class_id": int(cls_id),
            "precision": float(precision_per_class[i]) if i < len(precision_per_class) else 0.0,
            "recall": float(recall_per_class[i]) if i < len(recall_per_class) else 0.0,
            "f1_score": float(f1_per_class[i]) if i < len(f1_per_class) else 0.0,
            "support": int(np.sum(all_labels == cls_id))
        }
    
    # Compile results
    results = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "top3_accuracy": float(top3_correct),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "total_samples": len(all_labels),
        "avg_inference_time_ms": float(np.mean(inference_times) * 1000),
        "class_names": class_names
    }
    
    # Save results
    results_path = output_dir / "evaluation_report.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total test samples: {results['total_samples']}")
    print(f"Accuracy:           {accuracy * 100:.2f}%")
    print(f"Top-3 Accuracy:     {top3_correct * 100:.2f}%")
    print(f"Precision (macro):  {precision_macro:.4f}")
    print(f"Recall (macro):     {recall_macro:.4f}")
    print(f"F1 Score (macro):   {f1_macro:.4f}")
    print(f"Avg Inference Time: {np.mean(inference_times) * 1000:.2f} ms")
    
    print("\nPer-class metrics:")
    for cls_name, metrics in per_class_metrics.items():
        print(f"  {cls_name:40s} | P: {metrics['precision']:.3f} "
              f"| R: {metrics['recall']:.3f} | F1: {metrics['f1_score']:.3f} "
              f"| N: {metrics['support']}")
    
    print(f"\n[Evaluation] Report saved to: {results_path}")
    print(f"[Evaluation] Confusion matrix saved to: {plot_path}")
    
    return results


def print_detailed_classification_report(results: Dict[str, Any]):
    """Print a detailed classification report."""
    report = results.get('classification_report', {})
    
    print("\n" + "=" * 90)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 90)
    print(f"{'Class':<45} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 90)
    
    for class_name, metrics in report.items():
        if class_name in ('accuracy', 'macro avg', 'weighted avg'):
            print(f"{class_name:<45} ", end='')
        else:
            print(f"{class_name:<45} ", end='')
        
        print(f"{metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} "
              f"{metrics['f1-score']:>10.4f} "
              f"{metrics['support']:>10.0f}")
    
    print("=" * 90)


def analyze_class_imbalance(class_counts: Dict[str, int]) -> Dict[str, Any]:
    """
    Analyze class imbalance in the dataset.
    
    Returns:
        Dictionary with imbalance analysis
    """
    counts = list(class_counts.values())
    if not counts:
        return {"error": "No class counts provided"}
    
    min_count = min(counts)
    max_count = max(counts)
    total = sum(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    # Determine imbalance level
    if imbalance_ratio > 10:
        assessment = "highly_imbalanced"
    elif imbalance_ratio > 3:
        assessment = "moderately_imbalanced"
    else:
        assessment = "approximately_balanced"
    
    analysis = {
        "total_samples": total,
        "num_classes": len(counts),
        "min_count": min_count,
        "max_count": max_count,
        "imbalance_ratio": float(imbalance_ratio),
        "assessment": assessment,
        "class_distribution": class_counts,
        "minority_classes": [name for name, count in class_counts.items() if count == min_count],
        "majority_classes": [name for name, count in class_counts.items() if count == max_count],
        "recommended_strategy": "class_weighted_loss" if imbalance_ratio > 3 else "standard"
    }
    
    print(f"\n[Evaluation] Class imbalance analysis:")
    print(f"  Assessment: {assessment}")
    print(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    print(f"  Recommended strategy: {analysis['recommended_strategy']}")
    
    return analysis


if __name__ == "__main__":
    print("Evaluation module loaded successfully")
    print("Usage: from backend.ai.disease_detection.evaluate import evaluate_model")