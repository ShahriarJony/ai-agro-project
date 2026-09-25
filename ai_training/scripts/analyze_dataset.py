#!/usr/bin/env python3
"""
Phase 6/7/8: Analyse the extracted dataset.

Produces:
  - data/manifests/dataset_manifest.csv    one row per image
  - data/manifests/class_mapping.json      deterministic class index mapping
  - data/manifests/invalid_images.csv      unreadable / corrupt / unsupported files
  - data/manifests/duplicates.csv          exact + near duplicate groups
  - reports/dataset_analysis.json          machine-readable summary

The raw dataset is never modified. Nothing is deleted; problems are recorded.
"""

import io
import os
import sys
import csv
import json
import time
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image, ImageFile

# Truncated files should surface as errors rather than silent warnings.
ImageFile.LOAD_TRUNCATED_IMAGES = False

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from verify_kaggle_dataset import load_env_file, resolve_env_file  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MIN_DIMENSION = 16
MAX_DIMENSION = 20000


def locate_dataset_root():
    """Find the directory whose direct children are the class folders."""
    extracted = WORKSPACE_ROOT / "data" / "extracted"
    if not extracted.exists():
        return None

    best = None
    for candidate in extracted.rglob("*"):
        if not candidate.is_dir():
            continue
        children = [c for c in candidate.iterdir() if c.is_dir()]
        if not children:
            continue
        image_children = [
            c for c in children
            if any(f.suffix.lower() in IMAGE_EXTENSIONS for f in c.iterdir() if f.is_file())
        ]
        if len(image_children) >= 5 and len(image_children) == len(children):
            depth = len(candidate.relative_to(extracted).parts)
            if best is None or depth > best[0]:
                best = (depth, candidate)

    return best[1] if best else None


def build_class_mapping(class_dirs):
    """
    Build a deterministic class mapping.

    Classes are sorted by crop then condition so the ordering is stable and
    human-meaningful, independent of filesystem enumeration order.
    """
    names = sorted(d.name for d in class_dirs)

    def sort_key(name):
        crop, _, condition = name.partition("___")
        return (crop.lower(), condition.lower())

    ordered = sorted(names, key=sort_key)
    return {index: name for index, name in enumerate(ordered)}


def inspect_image(path_str):
    """
    Inspect one image. Returns a plain dict (picklable for ProcessPoolExecutor).

    Never raises; problems are reported through the `error` field.
    """
    path = Path(path_str)
    result = {
        "path": str(path),
        "exists": path.exists(),
        "readable": False,
        "width": 0,
        "height": 0,
        "channels": 0,
        "mode": "",
        "format": "",
        "file_size": 0,
        "file_hash": "",
        "phash": "",
        "error": "",
    }

    try:
        result["file_size"] = path.stat().st_size
    except Exception as exc:
        result["error"] = f"stat failed: {exc}"
        return result

    if result["file_size"] == 0:
        result["error"] = "zero-byte file"
        return result

    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        result["error"] = "unsupported extension"
        return result

    # Exact-content hash for duplicate detection.
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
        result["file_hash"] = hasher.hexdigest()
    except Exception as exc:
        result["error"] = f"read failed: {exc}"
        return result

    # Decode + verify header and dimensions.
    try:
        with Image.open(path) as image:
            image.load()
            result["width"], result["height"] = image.size
            result["mode"] = image.mode
            result["format"] = image.format or ""
            result["channels"] = len(image.getbands())
            result["readable"] = True

            # Perceptual hash for near-duplicate detection.
            try:
                import imagehash

                result["phash"] = str(imagehash.phash(image.convert("RGB")))
            except Exception:
                result["phash"] = ""
    except Exception as exc:
        result["error"] = f"decode failed: {type(exc).__name__}: {exc}"
        return result

    if result["width"] < MIN_DIMENSION or result["height"] < MIN_DIMENSION:
        result["error"] = f"suspicious dimensions {result['width']}x{result['height']}"
        result["readable"] = False
    elif result["width"] > MAX_DIMENSION or result["height"] > MAX_DIMENSION:
        result["error"] = f"suspicious dimensions {result['width']}x{result['height']}"
        result["readable"] = False

    return result


def find_near_duplicates(records, threshold=4, max_pairs_per_hash=200):
    """
    Group perceptual hashes and flag near-duplicate pairs.

    Hamming distance <= threshold is treated as a near duplicate. Comparison is
    limited per hash bucket so runtime stays bounded.
    """
    buckets = defaultdict(list)
    for record in records:
        if record["phash"]:
            buckets[record["phash"]].append(record)

    near = []
    for phash, group in buckets.items():
        if len(group) < 2:
            continue
        # Cap the number of pairwise comparisons inside one bucket.
        for i in range(len(group)):
            for j in range(i + 1, min(i + 1 + max_pairs_per_hash, len(group))):
                a, b = group[i], group[j]
                try:
                    distance = int(phash, 16) ^ int(b["phash"], 16)
                    distance = bin(distance).count("1")
                except ValueError:
                    continue
                if distance <= threshold:
                    near.append(
                        {
                            "path_a": a["path"],
                            "path_b": b["path"],
                            "phash": phash,
                            "hamming_distance": distance,
                            "class_a": a.get("class_name", ""),
                            "class_b": b.get("class_name", ""),
                        }
                    )
    return near


def main():
    print("=" * 70)
    print("PHASE 6/7/8 - DATASET ANALYSIS")
    print("=" * 70)

    dataset_root = locate_dataset_root()
    if dataset_root is None:
        print("[FAIL] Could not locate the class root inside data/extracted/")
        print("       Run download_dataset.py first.")
        return 1

    print(f"[OK] Dataset root: {dataset_root}")

    class_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    class_mapping = build_class_mapping(class_dirs)
    reverse_mapping = {name: index for index, name in class_mapping.items()}

    print(f"[OK] Classes: {len(class_mapping)}")
    for index, name in class_mapping.items():
        print(f"       {index:2d}. {name}")

    image_files = []
    for class_dir in class_dirs:
        for file in sorted(class_dir.iterdir()):
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                image_files.append((str(file), class_dir.name))

    print(f"\n[...] Inspecting {len(image_files)} images ...")
    started = time.time()

    workers = min(16, (os.cpu_count() or 4))
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        paths_only = [p for p, _ in image_files]
        for index, result in enumerate(pool.map(inspect_image, paths_only, chunksize=200), start=1):
            result["class_name"] = image_files[index - 1][1]
            result["class_id"] = reverse_mapping.get(result["class_name"], -1)
            records.append(result)
            if index % 5000 == 0:
                print(f"    ... inspected {index}/{len(image_files)}")

    elapsed = time.time() - started
    print(f"[OK] Inspection finished in {elapsed:.1f}s")

    valid = [r for r in records if r["readable"]]
    invalid = [r for r in records if not r["readable"]]
    print(f"[OK] Valid images  : {len(valid)}")
    print(f"[OK] Invalid images: {len(invalid)}")

    # Exact duplicates (identical bytes)
    hash_groups = defaultdict(list)
    for record in valid:
        if record["file_hash"]:
            hash_groups[record["file_hash"]].append(record)

    exact_duplicate_groups = []
    for file_hash, group in hash_groups.items():
        if len(group) < 2:
            continue
        exact_duplicate_groups.append(
            {
                "sha256": file_hash,
                "count": len(group),
                "paths": [g["path"] for g in group],
                "classes": sorted({g["class_name"] for g in group}),
            }
        )

    exact_pairs = sum(g["count"] - 1 for g in exact_duplicate_groups)
    print(f"[OK] Exact duplicate groups: {len(exact_duplicate_groups)} ({exact_pairs} redundant copies)")

    print("[...] Computing perceptual hashes for near-duplicate detection ...")
    near_duplicates = find_near_duplicates(valid)
    print(f"[OK] Near-duplicate pairs: {len(near_duplicates)}")

    # Class distribution
    class_counts = Counter(r["class_name"] for r in valid)
    counts = list(class_counts.values())
    min_count, max_count = min(counts), max(counts)
    imbalance_ratio = max_count / min_count if min_count else float("inf")

    if imbalance_ratio > 10:
        assessment = "highly imbalanced"
    elif imbalance_ratio > 3:
        assessment = "moderately imbalanced"
    else:
        assessment = "approximately balanced"

    print(f"\n[OK] Class imbalance: {assessment} (ratio {imbalance_ratio:.2f})")

    # Dimension statistics
    widths = [r["width"] for r in valid]
    heights = [r["height"] for r in valid]
    dim_counts = Counter(f"{r['width']}x{r['height']}" for r in valid)
    top_dims = dim_counts.most_common(5)
    formats = Counter(r["format"] for r in valid)
    modes = Counter(r["mode"] for r in valid)

    print(f"[OK] Dimensions: min {min(widths)}x{min(heights)}, max {max(widths)}x{max(heights)}")
    print(f"     Most common: {', '.join(f'{d} ({c})' for d, c in top_dims)}")

    # Split detection (single-directory dataset has no official split)
    split_markers = {"train", "test", "val", "valid", "validation"}
    has_split = any(part.lower() in split_markers for part in dataset_root.parts)
    print(f"[OK] Predefined train/val/test split: {'YES' if has_split else 'NO'}")

    # Cross-class leakage check for exact duplicates
    cross_class_dupes = [g for g in exact_duplicate_groups if len(g["classes"]) > 1]
    print(f"[OK] Cross-class exact duplicates: {len(cross_class_dupes)}")

    # ---- write manifests ----
    manifests = WORKSPACE_ROOT / "data" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    manifest_path = manifests / "dataset_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image_path",
                "class_name",
                "class_id",
                "original_label",
                "normalized_label",
                "width",
                "height",
                "channels",
                "file_size",
                "sha256",
                "phash",
                "is_valid",
                "error",
            ]
        )
        for record in sorted(records, key=lambda r: r["path"]):
            name = record["class_name"]
            writer.writerow(
                [
                    record["path"],
                    name,
                    record["class_id"],
                    name,
                    name.replace("___", " - ").replace("_", " "),
                    record["width"],
                    record["height"],
                    record["channels"],
                    record["file_size"],
                    record["file_hash"][:32],
                    record["phash"],
                    "true" if record["readable"] else "false",
                    record["error"],
                ]
            )
    print(f"\n[OK] Manifest   : {manifest_path} ({len(records)} rows)")

    invalid_path = manifests / "invalid_images.csv"
    with open(invalid_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "class_name", "reason", "error", "file_size"])
        for record in sorted(invalid, key=lambda r: r["path"]):
            writer.writerow(
                [record["path"], record["class_name"], "unreadable", record["error"], record["file_size"]]
            )
    print(f"[OK] Invalid    : {invalid_path} ({len(invalid)} rows)")

    dup_path = manifests / "duplicates.csv"
    with open(dup_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["duplicate_type", "path_a", "path_b", "class_a", "class_b", "hash", "hamming_distance"]
        )
        for group in exact_duplicate_groups:
            base = group["paths"][0]
            for other in group["paths"][1:]:
                writer.writerow(["exact", base, other, "", "", group["sha256"][:32], "0"])
        for pair in near_duplicates:
            writer.writerow(
                [
                    "near",
                    pair["path_a"],
                    pair["path_b"],
                    pair["class_a"],
                    pair["class_b"],
                    pair["phash"],
                    pair["hamming_distance"],
                ]
            )
    print(f"[OK] Duplicates : {dup_path}")

    mapping_path = manifests / "class_mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "num_classes": len(class_mapping),
                "mapping": {str(k): v for k, v in class_mapping.items()},
                "display_labels": {v: v.replace("___", " - ").replace("_", " ") for v in class_mapping.values()},
                "healthy_classes": [v for v in class_mapping.values() if "healthy" in v.lower()],
                "disease_classes": [v for v in class_mapping.values() if "healthy" not in v.lower()],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] Class map  : {mapping_path}")

    analysis = {
        "dataset_root": str(dataset_root),
        "total_images": len(records),
        "valid_images": len(valid),
        "invalid_images": len(invalid),
        "num_classes": len(class_mapping),
        "class_mapping": {str(k): v for k, v in class_mapping.items()},
        "class_counts": dict(sorted(class_counts.items(), key=lambda kv: -kv[1])),
        "healthy_classes": [v for v in class_mapping.values() if "healthy" in v.lower()],
        "disease_classes": [v for v in class_mapping.values() if "healthy" not in v.lower()],
        "imbalance": {
            "min_count": min_count,
            "max_count": max_count,
            "ratio": round(imbalance_ratio, 3),
            "assessment": assessment,
        },
        "duplicates": {
            "exact_groups": len(exact_duplicate_groups),
            "exact_redundant_copies": exact_pairs,
            "cross_class_exact_groups": len(cross_class_dupes),
            "near_pairs": len(near_duplicates),
        },
        "dimensions": {
            "min_width": min(widths),
            "max_width": max(widths),
            "min_height": min(heights),
            "max_height": max(heights),
            "most_common": dict(top_dims),
        },
        "formats": dict(formats),
        "color_modes": dict(modes),
        "predefined_split": has_split,
    }

    reports = WORKSPACE_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    analysis_path = reports / "dataset_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"[OK] Analysis   : {analysis_path}")

    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)
    print(f"{'Class':<54} {'Images':>8}")
    print("-" * 64)
    for name in sorted(class_counts, key=lambda n: -class_counts[n]):
        print(f"{name:<54} {class_counts[name]:>8d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
