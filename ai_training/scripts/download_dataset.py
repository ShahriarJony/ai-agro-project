#!/usr/bin/env python3
"""
Phase 5: Download the selected Kaggle plant disease dataset.

The archive lands in ai_training/data/raw/ (git-ignored) and is extracted to
ai_training/data/extracted/ (also git-ignored). Nothing multi-GB is committed.

Credentials come from environment variables or ai_training/.env and are never
printed in full.
"""

import io
import os
import sys
import json
import time
import zipfile
import shutil
import hashlib
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from verify_kaggle_dataset import load_env_file, resolve_env_file, get_kaggle_api, format_size  # noqa: E402

DEFAULT_DATASET = "karagwaanntreasure/plant-disease-detection"


def raw_dir():
    return WORKSPACE_ROOT / "data" / "raw"


def extracted_dir():
    return WORKSPACE_ROOT / "data" / "extracted"


def safe_extract(archive_path, destination):
    """
    Extract a zip while refusing path-traversal entries.

    Rejects absolute paths, drive letters, and any entry that would resolve
    outside `destination`.
    """
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        total = len(archive.infolist())
        for index, member in enumerate(archive.infolist(), start=1):
            name = member.filename
            if name.startswith("/") or name.startswith("\\") or ":" in name.split("/")[0][1:3]:
                raise ValueError(f"Unsafe absolute path in archive: {name}")
            target = (destination / name).resolve()
            try:
                target.relative_to(destination)
            except ValueError:
                raise ValueError(f"Path traversal detected: {name}") from None
            if index % 5000 == 0:
                print(f"    ... extracted {index}/{total} entries")
        archive.extractall(destination)

    return destination


def existing_archive(dataset_ref):
    """Return an already-downloaded archive for this dataset, if any."""
    raw = raw_dir()
    if not raw.exists():
        return None
    expected_stem = dataset_ref.replace("/", "-")
    for candidate in raw.glob(f"{expected_stem}*.zip"):
        if candidate.stat().st_size > 0:
            return candidate
    zips = sorted(raw.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def main():
    dataset_ref = os.environ.get("DISEASE_DATASET") or DEFAULT_DATASET
    if len(sys.argv) > 1:
        dataset_ref = sys.argv[1]

    print("=" * 70)
    print("PHASE 5 - KAGGLE DATASET DOWNLOAD")
    print("=" * 70)
    print(f"Dataset: {dataset_ref}")
    print(f"URL    : https://www.kaggle.com/datasets/{dataset_ref}\n")

    env_vars = load_env_file(resolve_env_file())
    username = env_vars.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USERNAME")
    key = env_vars.get("KAGGLE_KEY") or os.environ.get("KAGGLE_KEY")
    if not username or not key:
        print("[FAIL] Kaggle credentials not configured.")
        print("       Set KAGGLE_USERNAME and KAGGLE_KEY in ai_training/.env")
        return 1

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    raw = raw_dir()
    raw.mkdir(parents=True, exist_ok=True)

    archive = existing_archive(dataset_ref)
    if archive:
        print(f"[SKIP] Archive already present: {archive} ({format_size(archive.stat().st_size)})")
    else:
        try:
            handle = get_kaggle_api()
        except Exception as exc:
            print(f"[FAIL] Kaggle authentication failed: {exc}")
            return 1
        print(f"[OK] Kaggle authentication SUCCESS ({handle['style']} API)")

        api, style = handle["api"], handle["style"]
        print("[...] Downloading dataset archive (this can take a while)...")
        started = time.time()

        try:
            if style == "modern":
                api.dataset_download_files(dataset_ref, path=str(raw), unzip=False, force=True)
            else:
                api.dataset_download_files(dataset_ref, path=str(raw), unzip=False)
        except Exception as exc:
            if "429" in str(exc) or "Too Many Requests" in str(exc):
                print("[FAIL] Kaggle rate limited the download. Wait a few minutes and re-run.")
            else:
                print(f"[FAIL] Download failed: {exc}")
            return 1

        elapsed = time.time() - started
        archive = existing_archive(dataset_ref)
        if not archive:
            print("[FAIL] Download finished but no archive was found in data/raw/")
            return 1
        print(f"[OK] Download finished in {elapsed / 60:.1f} min")

    size = archive.stat().st_size
    print(f"[OK] Archive: {archive}")
    print(f"[OK] Size   : {format_size(size)} ({size} bytes)")

    # Integrity check
    if not zipfile.is_zipfile(archive):
        print("[FAIL] Downloaded file is not a valid zip archive (incomplete download?)")
        return 1
    with zipfile.ZipFile(archive, "r") as zf:
        bad = zf.testzip()
        entry_count = len(zf.infolist())
    if bad:
        print(f"[FAIL] Archive is corrupt at entry: {bad}")
        return 1
    print(f"[OK] Archive integrity verified ({entry_count} entries)")

    destination = extracted_dir() / archive.stem
    marker = destination / ".extraction_complete"

    if marker.exists():
        print(f"[SKIP] Already extracted to {destination}")
    else:
        print(f"\n[...] Extracting to {destination} ...")
        started = time.time()
        try:
            safe_extract(archive, destination)
        except Exception as exc:
            print(f"[FAIL] Extraction failed: {exc}")
            return 1
        marker.write_text("ok", encoding="utf-8")
        print(f"[OK] Extraction finished in {(time.time() - started) / 60:.1f} min")

    # Locate the class root (the directory whose children are class folders).
    images = list(destination.rglob("*.jpg")) + list(destination.rglob("*.jpeg")) + list(destination.rglob("*.png"))
    print(f"\n[OK] Extracted images found: {len(images)}")

    class_dirs = sorted({p.parent for p in images})
    print(f"[OK] Class folders found   : {len(class_dirs)}")
    for path in class_dirs:
        count = len(list(path.glob("*.jpg"))) + len(list(path.glob("*.jpeg"))) + len(list(path.glob("*.png")))
        print(f"       {path.name:<52} {count:>6d}")

    extracted_bytes = sum(f.stat().st_size for f in destination.rglob("*") if f.is_file())
    print(f"\n[OK] Extracted size: {format_size(extracted_bytes)}")

    summary = {
        "dataset_ref": dataset_ref,
        "url": f"https://www.kaggle.com/datasets/{dataset_ref}",
        "archive_path": str(archive),
        "archive_bytes": size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "archive_entries": entry_count,
        "extract_path": str(destination),
        "extracted_bytes": extracted_bytes,
        "image_count": len(images),
        "class_count": len(class_dirs),
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    reports = WORKSPACE_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[OK] Download summary written to {reports / 'download_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
