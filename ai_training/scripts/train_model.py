#!/usr/bin/env python3
"""
Phase 10-12: Train the MobileNetV2 plant disease classifier.

Reads the leakage-aware split manifest, trains with class-weighted loss to
handle the 21x class imbalance, tracks the best validation checkpoint, and
writes the production model plus metadata.

CPU-only environments are supported; epochs and batch size are configurable.
Nothing here fabricates metrics - everything written is measured during the run.
"""

import io
import os
import sys
import csv
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent

import cv2

cv2.setNumThreads(0)

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed every RNG that can affect the run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def build_transforms(train: bool):
    """Training gets mild augmentation; eval stays deterministic."""
    if train:
        return A.Compose(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.95, 1.05),
                    rotate=(-12, 12),
                    translate_percent=(-0.04, 0.04),
                    shear=(-4, 4),
                    p=0.5,
                ),
                A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.5),
                A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=14, val_shift_limit=10, p=0.25),
                A.GaussNoise(std_range=(0.02, 0.10), p=0.15),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class LeafDataset(Dataset):
    """Dataset backed by the split manifest."""

    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        try:
            image = cv2.imread(record["image_path"], cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("cv2.imread returned None")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = self.transform(image=image)["image"]
        except Exception:
            # Never crash a batch: fall back to a normalised zero image.
            tensor = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        return tensor, int(record["class_id"])


def load_split(manifest_path: Path, split_name: str):
    """Load all rows for one split."""
    rows = []
    with open(manifest_path, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == split_name:
                rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(num_classes: int, pretrained: bool = True):
    """MobileNetV2 with a replaced classifier head."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.mobilenet_v2(weights=weights)

    model = nn.Sequential()
    model.add_module("features", backbone.features)
    model.add_module("pool", nn.AdaptiveAvgPool2d(1))
    model.add_module("flatten", nn.Flatten())
    model.add_module("dropout", nn.Dropout(p=0.3))
    model.add_module("classifier", nn.Linear(backbone.last_channel, num_classes))
    model._is_agroai_wrapper = True
    return model


def forward_features(model, x):
    """Forward through a plain nn.Sequential MobileNetV2 wrapper."""
    return model(x)


# --------------------------------------------------------------------------
# Train / evaluate helpers
# --------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    """Run one epoch. Trains when `optimizer` is provided, otherwise evaluates."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_preds = []
    all_targets = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            if training:
                loss.backward()
                optimizer.step()
                # Step the scheduler only after the optimizer has updated.
                if scheduler is not None:
                    scheduler.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_seen += batch_size
            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(targets.detach().cpu().tolist())

    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
        "correct": total_correct,
        "seen": total_seen,
        "preds": all_preds,
        "targets": all_targets,
    }


def macro_f1(targets, preds, num_classes):
    """Macro F1 computed without sklearn (keeps evaluation self-contained)."""
    scores = []
    for class_index in range(num_classes):
        tp = sum(1 for t, p in zip(targets, preds) if t == class_index and p == class_index)
        fp = sum(1 for t, p in zip(targets, preds) if t != class_index and p == class_index)
        fn = sum(1 for t, p in zip(targets, preds) if t == class_index and p != class_index)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        scores.append(f1)
    return sum(scores) / max(len(scores), 1)


# --------------------------------------------------------------------------
# Main training routine
# --------------------------------------------------------------------------
def train(args):
    set_seed(args.seed)

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"[FAIL] Split manifest not found: {manifest}")
        return None

    train_rows = load_split(manifest, "train")
    val_rows = load_split(manifest, "validation")
    test_rows = load_split(manifest, "test")

    if not train_rows:
        print("[FAIL] No training rows found in the manifest")
        return None

    class_ids = sorted({int(r["class_id"]) for r in train_rows})
    num_classes = len(class_ids)
    class_names = {int(r["class_id"]): r["class_name"] for r in train_rows}

    print("=" * 70)
    print("PHASE 10-12 - MOBILENETV2 TRAINING")
    print("=" * 70)
    print(f"Train / Val / Test : {len(train_rows)} / {len(val_rows)} / {len(test_rows)}")
    print(f"Classes            : {num_classes}")
    print(f"Image size         : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Epochs             : {args.epochs}")
    print(f"Learning rate      : {args.learning_rate}")
    print(f"Seed               : {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Device             : CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device             : CPU ({os.cpu_count()} logical cores) - expect slow training")

    # Class weights from the TRAIN split only.
    train_counts = Counter(int(r["class_id"]) for r in train_rows)
    weights = torch.tensor(
        [len(train_rows) / (num_classes * max(train_counts.get(i, 1), 1)) for i in range(num_classes)],
        dtype=torch.float32,
    )
    print(
        f"Class weight range : {weights.min():.3f} .. {weights.max():.3f} "
        f"(min class {min(train_counts.values())}, max class {max(train_counts.values())})"
    )

    train_loader = DataLoader(
        LeafDataset(train_rows, build_transforms(True)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        LeafDataset(val_rows, build_transforms(False)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=args.workers > 0,
    )
    test_loader = (
        DataLoader(
            LeafDataset(test_rows, build_transforms(False)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
        )
        if test_rows
        else None
    )

    model = build_model(num_classes, pretrained=not args.no_pretrained).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters         : {total_params:,}")

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "plant_disease_model.pth"

    history = {
        "epochs": [],
        "config": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "image_size": IMAGE_SIZE,
            "num_classes": num_classes,
            "device": str(device),
            "pretrained": not args.no_pretrained,
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        },
    }

    best_val_accuracy = -1.0
    best_epoch = 0
    patience = 0
    print("\n" + "=" * 70)
    print("EPOCH LOG")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        lr_now = optimizer.param_groups[0]["lr"]

        train_stats = run_epoch(model, train_loader, criterion, device, optimizer, scheduler)
        val_stats = run_epoch(model, val_loader, criterion, device)

        val_f1 = macro_f1(val_stats["targets"], val_stats["preds"], num_classes)
        train_f1 = macro_f1(train_stats["targets"], train_stats["preds"], num_classes)
        elapsed = time.time() - started

        record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "train_macro_f1": train_f1,
            "val_loss": val_stats["loss"],
            "val_accuracy": val_stats["accuracy"],
            "val_macro_f1": val_f1,
            "learning_rate": lr_now,
            "seconds": elapsed,
        }
        history["epochs"].append(record)

        improved = val_stats["accuracy"] > best_val_accuracy
        if improved:
            best_val_accuracy = val_stats["accuracy"]
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": [class_names[i] for i in range(num_classes)],
                    "class_mapping": {str(i): class_names[i] for i in range(num_classes)},
                    "num_classes": num_classes,
                    "architecture": "mobilenet_v2",
                    "image_size": IMAGE_SIZE,
                    "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
                    "validation_accuracy": best_val_accuracy,
                    "epoch": epoch,
                },
                best_path,
            )
        else:
            patience += 1

        marker = "*" if improved else " "
        print(
            f"{marker} Epoch {epoch:>3}/{args.epochs}  "
            f"train loss {train_stats['loss']:.4f} acc {train_stats['accuracy'] * 100:6.2f}% f1 {train_f1:.4f}  |  "
            f"val loss {val_stats['loss']:.4f} acc {val_stats['accuracy'] * 100:6.2f}% f1 {val_f1:.4f}  |  "
            f"lr {lr_now:.2e}  {elapsed / 60:.1f} min"
        )

        # Checkpoint history after every epoch so a crash still leaves a record.
        (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if args.patience > 0 and patience >= args.patience:
            print(f"\n[OK] Early stopping: no validation improvement for {patience} epochs")
            break

    # ---- Reload the best checkpoint before final evaluation ----
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"\n[OK] Restored best checkpoint from epoch {checkpoint['epoch']}")

    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 70)

    test_stats = None
    if test_loader is not None:
        test_stats = run_epoch(model, test_loader, criterion, device)
        test_f1 = macro_f1(test_stats["targets"], test_stats["preds"], num_classes)
        print(f"Test accuracy     : {test_stats['accuracy'] * 100:.2f}%")
        print(f"Test macro F1     : {test_f1:.4f}")
        print(f"Test samples      : {test_stats['seen']}")
    else:
        print("[WARN] No test split available")

    per_class = {}
    if test_stats:
        for class_index in range(num_classes):
            tp = sum(
                1 for t, p in zip(test_stats["targets"], test_stats["preds"]) if t == class_index and p == class_index
            )
            fp = sum(
                1 for t, p in zip(test_stats["targets"], test_stats["preds"]) if t != class_index and p == class_index
            )
            fn = sum(
                1 for t, p in zip(test_stats["targets"], test_stats["preds"]) if t == class_index and p != class_index
            )
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            per_class[class_names[class_index]] = {
                "class_id": class_index,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": sum(1 for t in test_stats["targets"] if t == class_index),
            }

    confusion = None
    if test_stats:
        confusion = [[0] * num_classes for _ in range(num_classes)]
        for true_index, pred_index in zip(test_stats["targets"], test_stats["preds"]):
            confusion[true_index][pred_index] += 1

    metadata = {
        "model_name": "MobileNetV2",
        "architecture": "mobilenet_v2",
        "num_classes": num_classes,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "framework": "PyTorch",
        "torch_version": torch.__version__,
        "pretrained_backbone": not args.no_pretrained,
        "dataset": "kaggle:karagwaanntreasure/plant-disease-detection",
        "dataset_url": "https://www.kaggle.com/datasets/karagwaanntreasure/plant-disease-detection",
        "random_seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history["epochs"]),
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "loss": "CrossEntropyLoss with inverse-frequency class weights",
        "device": str(device),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_accuracy": best_val_accuracy,
        "test_accuracy": test_stats["accuracy"] if test_stats else None,
        "class_names": [class_names[i] for i in range(num_classes)],
        "version": "1.0.0",
    }

    (output_dir / "class_names.json").write_text(
        json.dumps({str(i): class_names[i] for i in range(num_classes)}, indent=2), encoding="utf-8"
    )
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    reports_dir = WORKSPACE_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    evaluation = {
        "test_accuracy": test_stats["accuracy"] if test_stats else None,
        "test_macro_f1": macro_f1(test_stats["targets"], test_stats["preds"], num_classes) if test_stats else None,
        "validation_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "class_names": [class_names[i] for i in range(num_classes)],
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (reports_dir / "evaluation_report.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    # Confusion matrix image
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if confusion:
            matrix = np.array(confusion)
            labels = [class_names[i] for i in range(num_classes)]
            fig, ax = plt.subplots(figsize=(max(10, num_classes * 0.55), max(9, num_classes * 0.5)))
            image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(range(num_classes))
            ax.set_yticks(range(num_classes))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("MobileNetV2 Plant Disease - Test Confusion Matrix")
            fig.tight_layout()
            fig.savefig(reports_dir / "confusion_matrix.png", dpi=150)
            plt.close(fig)
            print(f"[OK] Confusion matrix -> {reports_dir / 'confusion_matrix.png'}")
    except Exception as exc:
        print(f"[WARN] Could not render confusion matrix: {exc}")

    print(f"\n[OK] Model       -> {best_path}")
    print(f"[OK] Class names -> {output_dir / 'class_names.json'}")
    print(f"[OK] Metadata    -> {output_dir / 'model_metadata.json'}")
    print(f"[OK] History     -> {output_dir / 'training_history.json'}")
    print(f"[OK] Evaluation  -> {reports_dir / 'evaluation_report.json'}")

    return {"model_path": str(best_path), "metadata": metadata, "history": history, "evaluation": evaluation}


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 plant disease classifier")
    parser.add_argument("--manifest", default=str(WORKSPACE_ROOT / "data" / "manifests" / "dataset_split.csv"))
    parser.add_argument("--output-dir", default=str(WORKSPACE_ROOT / "models"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights")
    args = parser.parse_args()

    result = train(args)
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
