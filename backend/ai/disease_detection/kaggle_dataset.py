"""
Kaggle Dataset Access Module for Plant Disease Detection
Handles authentication, dataset search, download, and extraction via Kaggle API.
"""

import os
import zipfile
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class KaggleDataset:
    """Represents a Kaggle dataset with metadata."""
    title: str
    slug: str
    owner: str
    ref: str
    url: str
    size: Optional[str] = None
    file_count: Optional[int] = None
    last_updated: Optional[str] = None


class KaggleDatasetManager:
    """Manages Kaggle dataset operations for plant disease detection."""
    
    DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api = None
        
    def authenticate(self) -> bool:
        """Authenticate with Kaggle API using environment variables."""
        try:
            import kaggle
            from kaggle.api.kaggle_api_utilities import KaggleApi
            
            username = os.environ.get("KAGGLE_USERNAME")
            key = os.environ.get("KAGGLE_KEY")
            
            if not username or not key:
                print("[Kaggle] ERROR: KAGGLE_USERNAME and KAGGLE_KEY must be set in environment")
                return False
                
            # Set environment for kaggle library
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key
            
            self.api = KaggleApi()
            self.api.authenticate()
            print("[Kaggle] Authentication successful")
            return True
            
        except ImportError:
            print("[Kaggle] ERROR: kaggle library not installed. Run: pip install kaggle")
            return False
        except Exception as e:
            print(f"[Kaggle] Authentication failed: {e}")
            return False
    
    def search_datasets(self, query: str, max_results: int = 20) -> List[KaggleDataset]:
        """Search for datasets on Kaggle."""
        if not self.api:
            if not self.authenticate():
                return []
        
        try:
            datasets = self.api.dataset_list(query, sort_by="relevance", sort_dir="desc", page=1, page_size=max_results)
            results = []
            for d in datasets:
                results.append(KaggleDataset(
                    title=d.title,
                    slug=d.slug,
                    owner=d.ownerUsername,
                    ref=d.ref,
                    url=f"https://www.kaggle.com/{d.ownerUsername}/{d.slug}",
                    size=getattr(d, 'totalBytes', None),
                    file_count=getattr(d, 'fileCount', None),
                    last_updated=getattr(d, 'lastUpdated', None)
                ))
            print(f"[Kaggle] Found {len(results)} datasets for query: '{query}'")
            return results
            
        except Exception as e:
            print(f"[Kaggle] Search failed: {e}")
            return []
    
    def find_plant_disease_datasets(self) -> List[KaggleDataset]:
        """Search for plant disease detection datasets using multiple queries."""
        all_datasets = []
        queries = [
            "Plant Disease Detection",
            "New Plant Disease Detection",
            "plant disease",
            "leaf disease",
            "plantvillage",
            "crop disease"
        ]
        
        seen_refs = set()
        for query in queries:
            datasets = self.search_datasets(query, max_results=10)
            for ds in datasets:
                if ds.ref not in seen_refs:
                    seen_refs.add(ds.ref)
                    all_datasets.append(ds)
        
        # Filter for plant disease related
        filtered = []
        for ds in all_datasets:
            title_lower = ds.title.lower()
            if any(term in title_lower for term in ["plant disease", "leaf disease", "plantvillage", "crop disease", "foliar"]):
                filtered.append(ds)
        
        print(f"[Kaggle] Found {len(filtered)} relevant plant disease datasets")
        return filtered
    
    def download_dataset(self, dataset_ref: str) -> Optional[Path]:
        """Download a dataset by reference (owner/slug)."""
        if not self.api:
            if not self.authenticate():
                return None
        
        try:
            print(f"[Kaggle] Downloading dataset: {dataset_ref}")
            self.api.dataset_download_files(dataset_ref, path=str(self.cache_dir), unzip=False)
            
            zip_path = self.cache_dir / f"{dataset_ref.replace('/', '-')}.zip"
            if zip_path.exists():
                print(f"[Kaggle] Downloaded to: {zip_path}")
                return zip_path
            else:
                # Find the actual zip file
                zip_files = list(self.cache_dir.glob("*.zip"))
                if zip_files:
                    return zip_files[0]
                return None
                
        except Exception as e:
            print(f"[Kaggle] Download failed: {e}")
            return None
    
    def extract_dataset(self, zip_path: Path, extract_dir: Optional[Path] = None) -> Optional[Path]:
        """Extract a downloaded dataset zip file."""
        if extract_dir is None:
            extract_dir = self.cache_dir / "extracted" / zip_path.stem
        
        extract_dir = Path(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"[Kaggle] Extracting {zip_path} to {extract_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            print(f"[Kaggle] Extraction complete: {extract_dir}")
            return extract_dir
            
        except zipfile.BadZipFile:
            print(f"[Kaggle] ERROR: Invalid or corrupted zip file: {zip_path}")
            return None
        except Exception as e:
            print(f"[Kaggle] Extraction failed: {e}")
            return None
    
    def get_dataset_structure(self, dataset_path: Path) -> Dict:
        """Analyze the structure of an extracted dataset."""
        dataset_path = Path(dataset_path)
        structure = {
            "root": str(dataset_path),
            "directories": [],
            "files": [],
            "image_extensions": ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'],
            "classes": [],
            "class_counts": {},
            "total_images": 0
        }
        
        if not dataset_path.exists():
            return structure
        
        # Walk the directory
        for root, dirs, files in os.walk(dataset_path):
            rel_root = Path(root).relative_to(dataset_path)
            if rel_root != Path('.'):
                structure["directories"].append(str(rel_root))
            
            for file in files:
                rel_file = Path(root).relative_to(dataset_path) / file
                structure["files"].append(str(rel_file))
                
                # Check if image
                ext = Path(file).suffix.lower()
                if ext in structure["image_extensions"]:
                    structure["total_images"] += 1
                    # Try to infer class from parent directory
                    class_name = rel_root.name if rel_root != Path('.') else "unknown"
                    if class_name not in structure["class_counts"]:
                        structure["class_counts"][class_name] = 0
                        structure["classes"].append(class_name)
                    structure["class_counts"][class_name] += 1
        
        print(f"[Kaggle] Dataset structure: {structure['total_images']} images, {len(structure['classes'])} classes")
        return structure
    
    def save_dataset_manifest(self, structure: Dict, output_path: Path):
        """Save dataset structure as manifest JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(structure, f, indent=2)
        
        print(f"[Kaggle] Dataset manifest saved to: {output_path}")


def get_plant_disease_dataset(cache_dir: Optional[str] = None) -> Tuple[Optional[Path], Optional[Dict]]:
    """
    Convenience function to find, download, and extract a plant disease dataset.
    Returns (dataset_path, structure) or (None, None) on failure.
    """
    manager = KaggleDatasetManager(cache_dir)
    
    if not manager.authenticate():
        return None, None
    
    # Find datasets
    datasets = manager.find_plant_disease_datasets()
    
    if not datasets:
        print("[Kaggle] No suitable plant disease datasets found")
        return None, None
    
    # Select the most relevant dataset (first one)
    selected = datasets[0]
    print(f"[Kaggle] Selected dataset: {selected.title} ({selected.ref})")
    
    # Download
    zip_path = manager.download_dataset(selected.ref)
    if not zip_path:
        return None, None
    
    # Extract
    extract_dir = manager.extract_dataset(zip_path)
    if not extract_dir:
        return None, None
    
    # Analyze structure
    structure = manager.get_dataset_structure(extract_dir)
    
    # Save manifest
    manifest_path = manager.cache_dir / "dataset_manifest.json"
    manager.save_dataset_manifest(structure, manifest_path)
    
    return extract_dir, structure


if __name__ == "__main__":
    # Test the module
    print("Testing Kaggle Dataset Manager...")
    path, structure = get_plant_disease_dataset()
    if path:
        print(f"Dataset ready at: {path}")
        print(f"Classes: {structure.get('classes', [])}")
        print(f"Class counts: {structure.get('class_counts', {})}")
    else:
        print("Failed to get dataset")