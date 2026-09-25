"""
Dataset handling module for Plant Disease Detection
Provides dataset loading, validation, cleaning, and class mapping.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import Counter
import cv2
import numpy as np
from PIL import Image


@dataclass
class ImageRecord:
    """Represents a single image in the dataset."""
    path: str
    class_name: str
    class_id: int
    width: int
    height: int
    channels: int
    file_size: int
    is_valid: bool
    error: Optional[str] = None
    file_hash: Optional[str] = None


@dataclass
class DatasetManifest:
    """Complete dataset manifest with all image records and metadata."""
    total_images: int
    valid_images: int
    invalid_images: int
    classes: List[str]
    class_counts: Dict[str, int]
    class_mapping: Dict[int, str]
    image_records: List[ImageRecord]
    duplicates: List[Tuple[str, str]]  # (original, duplicate)
    corrupted_files: List[str]


class PlantDiseaseDataset:
    """Handles plant disease dataset loading, validation, and analysis."""
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    DEFAULT_IMAGE_SIZE = (224, 224)
    
    def __init__(self, dataset_path: str, class_mapping_path: Optional[str] = None):
        self.dataset_path = Path(dataset_path)
        self.class_mapping_path = Path(class_mapping_path) if class_mapping_path else None
        self.class_mapping: Dict[int, str] = {}
        self.reverse_mapping: Dict[str, int] = {}
        self.manifest: Optional[DatasetManifest] = None
        
    def _load_class_mapping(self) -> bool:
        """Load existing class mapping from file."""
        if self.class_mapping_path and self.class_mapping_path.exists():
            try:
                with open(self.class_mapping_path, 'r') as f:
                    data = json.load(f)
                    # Handle both formats: {"0": "class"} or [{"id": 0, "name": "class"}]
                    if isinstance(data, dict):
                        self.class_mapping = {int(k): v for k, v in data.items()}
                    elif isinstance(data, list):
                        self.class_mapping = {item['id']: item['name'] for item in data}
                    self.reverse_mapping = {v: k for k, v in self.class_mapping.items()}
                    return True
            except Exception as e:
                print(f"[Dataset] Warning: Failed to load class mapping: {e}")
        return False
    
    def _save_class_mapping(self) -> bool:
        """Save class mapping to file."""
        if self.class_mapping_path:
            try:
                self.class_mapping_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.class_mapping_path, 'w') as f:
                    json.dump(self.class_mapping, f, indent=2)
                return True
            except Exception as e:
                print(f"[Dataset] Warning: Failed to save class mapping: {e}")
        return False
    
    def discover_classes(self) -> List[str]:
        """Discover classes from directory structure."""
        classes = []
        if self.dataset_path.exists():
            for item in sorted(self.dataset_path.iterdir()):
                if item.is_dir():
                    classes.append(item.name)
        return classes
    
    def build_class_mapping(self, classes: Optional[List[str]] = None) -> Dict[int, str]:
        """Build class ID to name mapping."""
        if classes is None:
            classes = self.discover_classes()
        
        # Sort classes for consistent ordering
        classes = sorted(classes)
        
        self.class_mapping = {i: name for i, name in enumerate(classes)}
        self.reverse_mapping = {name: i for i, name in enumerate(classes)}
        self._save_class_mapping()
        
        print(f"[Dataset] Built class mapping with {len(classes)} classes")
        return self.class_mapping
    
    def get_class_id(self, class_name: str) -> Optional[int]:
        """Get class ID from class name."""
        return self.reverse_mapping.get(class_name)
    
    def get_class_name(self, class_id: int) -> Optional[str]:
        """Get class name from class ID."""
        return self.class_mapping.get(class_id)
    
    def validate_image(self, image_path: Path) -> Tuple[bool, Optional[str], Optional[Tuple[int, int, int]], Optional[int]]:
        """Validate a single image file."""
        try:
            # Check file exists and is readable
            if not image_path.exists():
                return False, "File does not exist", None, None
            
            file_size = image_path.stat().st_size
            if file_size == 0:
                return False, "Zero-byte file", None, None
            
            # Try to open with PIL (handles more formats)
            with Image.open(image_path) as img:
                img.verify()  # Verify it's a valid image
                
            # Reopen for actual processing (verify closes the file)
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                width, height = img.size
                channels = len(img.getbands())
            
            # Additional validation with OpenCV
            cv_img = cv2.imread(str(image_path))
            if cv_img is None:
                return False, "OpenCV cannot read image", None, None
            
            if cv_img.shape[0] == 0 or cv_img.shape[1] == 0:
                return False, "Invalid dimensions", None, None
            
            return True, None, (width, height, channels), file_size
            
        except Exception as e:
            return False, str(e), None, None
    
    def compute_file_hash(self, image_path: Path) -> str:
        """Compute SHA256 hash of file for duplicate detection."""
        hasher = hashlib.sha256()
        try:
            with open(image_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""
    
    def scan_dataset(self, max_images: Optional[int] = None) -> DatasetManifest:
        """Scan entire dataset and build manifest."""
        print(f"[Dataset] Scanning dataset at: {self.dataset_path}")
        
        # Load or build class mapping
        if not self._load_class_mapping():
            self.build_class_mapping()
        
        image_records = []
        class_counts = Counter()
        hash_to_path = {}
        duplicates = []
        corrupted_files = []
        total_scanned = 0
        
        # Walk through dataset
        for class_dir in sorted(self.dataset_path.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            class_id = self.get_class_id(class_name)
            if class_id is None:
                print(f"[Dataset] Warning: Class '{class_name}' not in mapping, skipping")
                continue
            
            image_files = [f for f in class_dir.iterdir() 
                          if f.suffix.lower() in self.SUPPORTED_EXTENSIONS]
            
            for img_path in image_files:
                if max_images and total_scanned >= max_images:
                    break
                
                total_scanned += 1
                
                # Validate image
                is_valid, error, dims, file_size = self.validate_image(img_path)
                
                if is_valid and dims:
                    width, height, channels = dims
                    
                    # Compute hash for duplicate detection
                    file_hash = self.compute_file_hash(img_path)
                    
                    # Check for duplicates
                    if file_hash in hash_to_path:
                        duplicates.append((hash_to_path[file_hash], str(img_path)))
                    else:
                        hash_to_path[file_hash] = str(img_path)
                    
                    record = ImageRecord(
                        path=str(img_path),
                        class_name=class_name,
                        class_id=class_id,
                        width=width,
                        height=height,
                        channels=channels,
                        file_size=file_size,
                        is_valid=True,
                        file_hash=file_hash
                    )
                    class_counts[class_name] += 1
                else:
                    record = ImageRecord(
                        path=str(img_path),
                        class_name=class_name,
                        class_id=class_id,
                        width=0,
                        height=0,
                        channels=0,
                        file_size=file_size or 0,
                        is_valid=False,
                        error=error
                    )
                    corrupted_files.append(str(img_path))
                
                image_records.append(record)
            
            if max_images and total_scanned >= max_images:
                break
        
        valid_count = sum(1 for r in image_records if r.is_valid)
        invalid_count = len(image_records) - valid_count
        
        self.manifest = DatasetManifest(
            total_images=len(image_records),
            valid_images=valid_count,
            invalid_images=invalid_count,
            classes=list(self.class_mapping.values()),
            class_counts=dict(class_counts),
            class_mapping=self.class_mapping.copy(),
            image_records=image_records,
            duplicates=duplicates,
            corrupted_files=corrupted_files
        )
        
        print(f"[Dataset] Scan complete: {valid_count} valid, {invalid_count} invalid, {len(duplicates)} duplicates")
        return self.manifest
    
    def get_train_val_test_split(self, 
                                  train_ratio: float = 0.8, 
                                  val_ratio: float = 0.1, 
                                  test_ratio: float = 0.1,
                                  seed: int = 42) -> Tuple[List[ImageRecord], List[ImageRecord], List[ImageRecord]]:
        """Split dataset into train/val/test sets."""
        if not self.manifest:
            self.scan_dataset()
        
        valid_records = [r for r in self.manifest.image_records if r.is_valid]
        
        # Group by class for stratified split
        class_groups = {}
        for record in valid_records:
            if record.class_name not in class_groups:
                class_groups[record.class_name] = []
            class_groups[record.class_name].append(record)
        
        train_set = []
        val_set = []
        test_set = []
        
        np.random.seed(seed)
        
        for class_name, records in class_groups.items():
            np.random.shuffle(records)
            n = len(records)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train_set.extend(records[:n_train])
            val_set.extend(records[n_train:n_train + n_val])
            test_set.extend(records[n_train + n_val:])
        
        print(f"[Dataset] Split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
        return train_set, val_set, test_set
    
    def save_manifest(self, output_path: str):
        """Save manifest to JSON file."""
        if not self.manifest:
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = {
            "total_images": self.manifest.total_images,
            "valid_images": self.manifest.valid_images,
            "invalid_images": self.manifest.invalid_images,
            "classes": self.manifest.classes,
            "class_counts": self.manifest.class_counts,
            "class_mapping": self.manifest.class_mapping,
            "duplicates": self.manifest.duplicates,
            "corrupted_files": self.manifest.corrupted_files,
            "image_records": [asdict(r) for r in self.manifest.image_records]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[Dataset] Manifest saved to: {output_path}")
    
    def print_summary(self):
        """Print dataset summary."""
        if not self.manifest:
            return
        
        m = self.manifest
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total images:     {m.total_images}")
        print(f"Valid images:     {m.valid_images}")
        print(f"Invalid images:   {m.invalid_images}")
        print(f"Classes:          {len(m.classes)}")
        print(f"Duplicates:       {len(m.duplicates)}")
        print(f"Corrupted files:  {len(m.corrupted_files)}")
        print("\nClass distribution:")
        for class_name in sorted(m.class_counts.keys()):
            count = m.class_counts[class_name]
            pct = (count / m.valid_images * 100) if m.valid_images > 0 else 0
            print(f"  {class_name:30s} : {count:5d} ({pct:5.1f}%)")
        
        # Class imbalance check
        if m.class_counts:
            counts = list(m.class_counts.values())
            min_count = min(counts)
            max_count = max(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            print(f"\nClass imbalance ratio: {imbalance_ratio:.2f} (max/min)")
            if imbalance_ratio > 10:
                print("  WARNING: Severe class imbalance detected")
            elif imbalance_ratio > 3:
                print("  WARNING: Moderate class imbalance detected")


def create_dataset_from_kaggle(dataset_path: str, 
                                class_mapping_path: Optional[str] = None,
                                max_images: Optional[int] = None) -> PlantDiseaseDataset:
    """Factory function to create and scan a dataset."""
    dataset = PlantDiseaseDataset(dataset_path, class_mapping_path)
    dataset.scan_dataset(max_images=max_images)
    dataset.print_summary()
    return dataset


if __name__ == "__main__":
    # Test with sample path
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/test"
    if Path(test_path).exists():
        ds = create_dataset_from_kaggle(test_path)
    else:
        print(f"Test path {test_path} does not exist")