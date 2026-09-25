#!/usr/bin/env python3
"""
Phase 9: Create a reproducible, leakage-aware train/validation/test split.

Reads the manifest produced by analyze_dataset.py and writes split assignments.

Leakage control: images that are exact duplicates (same sha256) or near
duplicates (perceptual hash within Hamming distance <= NEAR_THRESHOLD) are
placed in the SAME split via union-find grouping, so a visually identical
image can never appear in both train and test.

No image is copied, resized or deleted. Only a split label is assigned.
"""

import io
import os
import sys
import csv
import json
import random
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent

MANIFEST = WORKSPACE_ROOT / "data" / "manifests" / "dataset_manifest.csv"
DUPLICATES = WORKSPACE_ROOT / "data" / "manifests" / "duplicates.csv"
OUT_MANIFEST = WORKSPACE_ROOT / "data" / "manifests" / "dataset_split.csv"
OUT_SUMMARY = WORKSPACE_ROOT / "reports" / "split_summary.json"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42
NEAR_THRESHOLD = 4


class UnionFind:
    """Minimal union-find for grouping related images."""

    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def load_manifest():
    rows = []
    with open(MANIFEST, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["is_valid"].lower() == "true":
                rows.append(row)
    return rows


def load_duplicate_groups():
    """Return list of (path_a, path_b) pairs that must stay in the same split."""
    pairs = []
    if not DUPLICATES.exists():
        return pairs
    with open(DUPLICATES, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row.get("hamming_distance") or 0) <= NEAR_THRESHOLD:
                pairs.append((row["path_a"], row["path_b"]))
    return pairs


def main():
    print("=" * 70)
    print("PHASE 9 - STRATIFIED, LEAKAGE-AWARE SPLIT")
    print("=" * 70)

    if not MANIFEST.exists():
        print(f"[FAIL] Manifest not found: {MANIFEST}")
        print("       Run analyze_dataset.py first.")
        return 1

    rows = load_manifest()
    print(f"[OK] Valid images loaded from manifest: {len(rows)}")

    # Group related images so duplicates stay together.
    union = UnionFind()
    for row in rows:
        union.find(row["image_path"])

    pairs = load_duplicate_groups()
    for path_a, path_b in pairs:
        if path_a in union.parent and path_b in union.parent:
            union.union(path_a, path_b)
    print(f"[OK] Duplicate/near-duplicate pairs linked: {len(pairs)}")

    groups = defaultdict(list)
    for row in rows:
        groups[union.find(row["image_path"])].append(row)
    print(f"[OK] Independent groups after grouping: {len(groups)}")

    # Stratify by class, splitting whole groups.
    by_class = defaultdict(list)
    for group in groups.values():
        class_name = group[0]["class_name"]
        by_class[class_name].append(group)

    rng = random.Random(SEED)
    assignments = {}
    split_counts = defaultdict(lambda: defaultdict(int))

    for class_name in sorted(by_class):
        class_groups = by_class[class_name][:]
        rng.shuffle(class_groups)

        total = sum(len(g) for g in class_groups)
        n_train = int(total * TRAIN_RATIO)
        n_val = int(total * VAL_RATIO)

        train_groups, val_groups, test_groups = [], [], []
        running = 0

        for group in class_groups:
            size = len(group)
            if running + size <= n_train:
                train_groups.append(group)
                running += size
            elif running < n_train + n_val:
                val_groups.append(group)
                running += size
            else:
                test_groups.append(group)

        # Backfill in case rounding left train short.
        for group in test_groups[:]:
            if running < n_train:
                train_groups.append(group)
                test_groups.remove(group)
                running += len(group)

        for name, group_list in (("train", train_groups), ("validation", val_groups), ("test", test_groups)):
            for group in group_list:
                for row in group:
                    assignments[row["image_path"]] = name
                    split_counts[name][class_name] += 1

    for name in ("train", "validation", "test"):
        total = sum(split_counts[name].values())
        print(f"[OK] {name:<11}: {total:>6d} images")

    # Leakage verification.
    group_splits = defaultdict(set)
    for path, split in assignments.items():
        group_splits[union.find(path)].add(split)
    leaked = [g for g, splits in group_splits.items() if len(splits) > 1]
    print(f"[OK] Groups spanning multiple splits: {len(leaked)} (must be 0)")

    # Write split manifest.
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ["split"]
    with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "split": assignments[row["image_path"]]})

    summary = {
        "seed": SEED,
        "ratios": {"train": TRAIN_RATIO, "validation": VAL_RATIO, "test": TEST_RATIO},
        "near_duplicate_hamming_threshold": NEAR_THRESHOLD,
        "total_images": len(rows),
        "independent_groups": len(groups),
        "linked_pairs": len(pairs),
        "groups_spanning_multiple_splits": len(leaked),
        "per_split_per_class": {k: dict(v) for k, v in split_counts.items()},
        "per_split_totals": {k: sum(v.values()) for k, v in split_counts.items()},
    }
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[OK] Split manifest: {OUT_MANIFEST}")
    print(f"[OK] Split summary : {OUT_SUMMARY}")

    print("\n" + "=" * 70)
    print("PER-CLASS SPLIT DISTRIBUTION")
    print("=" * 70)
    print(f"{'Class':<52} {'Train':>7} {'Val':>6} {'Test':>6}")
    print("-" * 74)
    for class_name in sorted(by_class):
        print(
            f"{class_name:<52} "
            f"{split_counts['train'][class_name]:>7} "
            f"{split_counts['validation'][class_name]:>6} "
            f"{split_counts['test'][class_name]:>6}"
        )

    if leaked:
        print("\n[FAIL] Data leakage detected: some duplicate groups span splits")
        return 1

    print("\n[OK] No duplicate group spans more than one split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
