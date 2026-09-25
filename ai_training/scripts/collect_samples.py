#!/usr/bin/env python3
"""
Collect real leaf images for end-to-end disease detection testing.

Copies a small, balanced sample of genuine dataset images into
ai_training/data/samples/ (git-ignored) so the FastAPI endpoint and the
frontend can be exercised with real photos instead of synthetic blobs.

Also writes a samples/index.json describing what each file actually contains,
which lets tests assert against ground truth rather than guessing.
"""

import io
import os
import sys
import json
import shutil
import random
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent

MANIFEST = WORKSPACE_ROOT / "data" / "manifests" / "dataset_split.csv"
SAMPLES_DIR = WORKSPACE_ROOT / "data" / "samples"
INDEX_PATH = SAMPLES_DIR / "index.json"

# Two per class keeps the sample set small but covers every class.
PER_CLASS = 2
SEED = 12345

# A handful of edge-case files for error-handling tests.
INVALID_TEXT = b"This is not an image. It is plain text pretending to be a .jpg file.\n"


def slug(class_name: str) -> str:
    """Filesystem-safe form of a class label."""
    return class_name.replace("(", "").replace(")", "").replace(" ", "_").replace("/", "_")


def main():
    print("=" * 70)
    print("SAMPLE IMAGE COLLECTION")
    print("=" * 70)

    if not MANIFEST.exists():
        print(f"[FAIL] Manifest not found: {MANIFEST}")
        print("       Run analyze_dataset.py and create_splits.py first.")
        return 1

    import csv

    by_class = {}
    with open(MANIFEST, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["is_valid"].lower() != "true":
                continue
            by_class.setdefault(row["class_name"], []).append(row)

    print(f"[OK] Classes available: {len(by_class)}")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    index = {
        "description": "Real images copied from the Kaggle plant disease dataset for E2E testing.",
        "dataset": "kaggle:karagwaanntreasure/plant-disease-detection",
        "per_class": PER_CLASS,
        "seed": SEED,
        "samples": [],
        "invalid_samples": [],
    }

    rng = random.Random(SEED)
    copied = 0

    for class_name in sorted(by_class):
        rows = by_class[class_name]
        # Prefer test-split rows so samples are genuinely unseen images.
        test_rows = [r for r in rows if r["split"] == "test"]
        pool = test_rows or rows
        chosen = rng.sample(pool, min(PER_CLASS, len(pool)))

        for row in chosen:
            source = Path(row["image_path"])
            if not source.exists():
                continue
            target = SAMPLES_DIR / f"{slug(class_name)}__{source.name}"
            try:
                shutil.copy2(source, target)
            except Exception as exc:
                print(f"[WARN] Could not copy {source.name}: {exc}")
                continue

            index["samples"].append(
                {
                    "file": target.name,
                    "class_id": int(row["class_id"]),
                    "class_name": class_name,
                    "expected_label": class_name,
                    "is_healthy": "healthy" in class_name.lower(),
                    "source_split": row["split"],
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "bytes": target.stat().st_size,
                }
            )
            copied += 1

    print(f"[OK] Real sample images copied: {copied}")

    # ---- edge-case files for API error handling tests ----
    edge_cases = [
        ("invalid_text.jpg", INVALID_TEXT, "not an image"),
        ("empty.jpg", b"", "zero bytes"),
    ]
    for filename, payload, reason in edge_cases:
        path = SAMPLES_DIR / filename
        path.write_bytes(payload)
        index["invalid_samples"].append({"file": filename, "reason": reason, "bytes": len(payload)})
        print(f"[OK] Edge-case file written: {filename} ({reason})")

    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")

    healthy = [s for s in index["samples"] if s["is_healthy"]]
    diseased = [s for s in index["samples"] if not s["is_healthy"]]

    print(f"\n[OK] Samples directory: {SAMPLES_DIR}")
    print(f"[OK] Total real samples: {len(index['samples'])}")
    print(f"     healthy classes : {len(healthy)} images")
    print(f"     diseased classes: {len(diseased)} images")
    print(f"     invalid samples : {len(index['invalid_samples'])}")
    print(f"[OK] Index written to  : {INDEX_PATH}")

    print("\nSAMPLE LISTING")
    print("-" * 70)
    for sample in index["samples"][:10]:
        print(f"  {sample['file'][:52]:<52} {sample['class_name']}")
    if len(index["samples"]) > 10:
        print(f"  ... and {len(index['samples']) - 10} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
