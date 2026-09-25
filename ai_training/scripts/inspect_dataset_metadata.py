#!/usr/bin/env python3
"""
Phase 3: Inspect Kaggle dataset metadata and full file listing without downloading.

Only metadata is fetched - no dataset archive is downloaded.
Supports kaggle 1.x and kaggle 2.x. Never prints credentials.
"""

import io
import os
import sys
import json
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_kaggle_dataset import load_env_file, resolve_env_file, get_kaggle_api, format_size  # noqa: E402

DEFAULT_TARGETS = [
    "karagwaanntreasure/plant-disease-detection",
    "vipoooool/new-plant-diseases-dataset",
]


def ref_to_dirname(dataset_ref):
    """Convert an owner/slug ref into a safe directory name."""
    return dataset_ref.replace("/", "-")


def get_metadata(api, style, dataset_ref, workdir):
    """
    Fetch dataset metadata across API versions.

    kaggle 2.x writes `dataset-metadata.json` into `workdir` and returns the
    path, so the JSON is read back from disk. The payload nests real fields
    under an "info" key.
    """
    if style == "modern":
        target_dir = Path(workdir) / ref_to_dirname(dataset_ref)
        target_dir.mkdir(parents=True, exist_ok=True)
        written = api.dataset_metadata(dataset_ref, str(target_dir))
        meta_path = Path(written) if written else target_dir / "dataset-metadata.json"
        if meta_path.is_dir():
            meta_path = meta_path / "dataset-metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload.get("info", payload)
        return {}

    view = api.dataset_view(dataset_ref)
    return {
        "title": getattr(view, "title", ""),
        "ownerUser": getattr(view, "ownerUser", ""),
        "datasetSlug": getattr(view, "datasetSlug", ""),
        "subtitle": getattr(view, "subtitle", ""),
        "description": getattr(view, "description", ""),
        "totalBytes": getattr(view, "totalBytes", 0),
        "fileCount": getattr(view, "fileCount", None),
        "lastUpdated": getattr(view, "lastUpdated", ""),
        "usabilityRating": getattr(view, "usabilityRating", None),
        "totalVotes": getattr(view, "totalVotes", None),
        "totalViews": getattr(view, "totalViews", None),
        "licenses": [{"name": getattr(view, "licenseName", "")}],
    }


def list_files(api, style, dataset_ref, max_pages=2000, page_size=200, progress_every=50, cache_path=None):
    """
    List every file in a dataset using resumable pagination.

    kaggle 2.x returns ApiListDatasetFilesResponse with .files/.next_page_token
    and caps page_size at 200. kaggle 1.x returns a flat list per page.

    Kaggle rate-limits aggressive pagination (HTTP 429), so this retries with
    exponential backoff and checkpoints partial results to `cache_path`.
    """
    files = []
    pages = 0
    token = None

    # Resume from a previous partial run when a cache file exists.
    if cache_path and Path(cache_path).exists():
        try:
            cached = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            files = cached.get("files", [])
            token = cached.get("next_page_token")
            pages = cached.get("pages", 0)
            if files:
                print(f"    resuming from cache: {len(files)} files, {pages} pages")
        except Exception:
            files, token, pages = [], None, 0

    def checkpoint():
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(
                json.dumps({"files": files, "next_page_token": token, "pages": pages}),
                encoding="utf-8",
            )

    def fetch():
        if style == "modern":
            response = api.dataset_list_files(dataset_ref, page_token=token, page_size=page_size)
            batch = list(getattr(response, "files", None) or [])
            return batch, getattr(response, "next_page_token", None)
        batch = api.dataset_list_files(dataset_ref, page=pages + 1)
        return list(batch or []), None

    def as_records(batch):
        """Normalize SDK objects into plain JSON-serializable dicts."""
        records = []
        for record in batch:
            if isinstance(record, dict):
                records.append(record)
            else:
                records.append(
                    {
                        "name": getattr(record, "name", ""),
                        "totalBytes": getattr(record, "total_bytes", 0) or 0,
                        "ref": getattr(record, "ref", ""),
                    }
                )
        return records

    while pages < max_pages:
        try:
            batch, next_token = fetch()
        except Exception as exc:
            if "429" in str(exc) or "Too Many Requests" in str(exc):
                delay = min(60, 2 ** min(pages // 20, 5))
                print(f"    rate limited, backing off {delay}s ...")
                checkpoint()
                time.sleep(delay)
                continue
            checkpoint()
            print(f"    [WARN] listing stopped: {exc}")
            break

        files.extend(as_records(batch))
        pages += 1
        token = next_token

        if pages % progress_every == 0:
            print(f"    ... paginated {len(files)} files ({pages} pages)")
            checkpoint()

        if not batch or not token:
            break

    checkpoint()
    return files


def analyze_listing(files):
    """Derive structure statistics from a Kaggle file listing."""
    extensions = {}
    directories = {}
    total_bytes = 0
    image_dirs = set()

    for record in files:
        if isinstance(record, dict):
            name = record.get("name", "")
            size = record.get("totalBytes", 0) or 0
        else:
            name = getattr(record, "name", "")
            size = getattr(record, "total_bytes", 0) or 0

        total_bytes += int(size or 0)

        suffix = Path(name).suffix.lower()
        if suffix:
            extensions[suffix] = extensions.get(suffix, 0) + 1

        parent = str(Path(name).parent).replace("\\", "/")
        if parent and parent != ".":
            directories[parent] = directories.get(parent, 0) + 1
            if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
                image_dirs.add(parent)

    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "extensions": dict(sorted(extensions.items(), key=lambda kv: -kv[1])),
        "image_directory_count": len(image_dirs),
        "directories": dict(sorted(directories.items(), key=lambda kv: -kv[1])),
    }


def detect_split_and_class(path):
    """
    Infer split and class from a dataset-relative image path.

    Handles layouts like:
      Dataset/<Class>/file.jpg
      Root/train/<Class>/file.jpg
      train/<Class>/file.jpg
    """
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    if not parts:
        return None, None

    split = None
    class_name = None

    for token in ("train", "valid", "validation", "val", "test"):
        if token in parts[:-1]:
            idx = parts.index(token)
            split = {"val": "validation", "valid": "validation"}.get(token, token)
            remainder = parts[idx + 1 : -1]
            if remainder:
                class_name = remainder[-1]
            break

    if class_name is None:
        class_name = parts[-2] if len(parts) >= 2 else None

    return split, class_name


def main():
    workspace_root = Path(__file__).resolve().parent.parent
    env_vars = load_env_file(resolve_env_file())

    username = env_vars.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USERNAME")
    key = env_vars.get("KAGGLE_KEY") or os.environ.get("KAGGLE_KEY")
    if not username or not key:
        print("[FAIL] Kaggle credentials not configured")
        return 1

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    targets = sys.argv[1:] or DEFAULT_TARGETS
    handle = get_kaggle_api()
    api, style = handle["api"], handle["style"]
    print(f"[OK] Kaggle authentication SUCCESS ({style} API)\n")

    workdir = workspace_root / "data" / "metadata_cache"
    workdir.mkdir(parents=True, exist_ok=True)

    results = []
    for ref in targets:
        print("=" * 70)
        print(f"DATASET: {ref}")
        print("=" * 70)

        try:
            meta = get_metadata(api, style, ref, workdir)
        except Exception as exc:
            print(f"[FAIL] Metadata request failed: {exc}")
            results.append({"ref": ref, "error": str(exc)})
            continue

        licenses = meta.get("licenses") or []
        license_names = [item.get("name", "") for item in licenses if isinstance(item, dict)]

        print(f"Title          : {meta.get('title', '')}")
        print(f"Owner          : {meta.get('ownerUser', '')}")
        print(f"Slug           : {meta.get('datasetSlug', '')}")
        print(f"Subtitle       : {meta.get('subtitle', '')}")
        print(f"Last updated   : {meta.get('lastUpdated', 'unknown')}")
        print(f"License        : {', '.join(n for n in license_names if n) or 'unknown'}")
        print(f"Usability      : {meta.get('usabilityRating')}")
        print(f"Votes / Views  : {meta.get('totalVotes')} / {meta.get('totalViews')}")

        print("\nFetching complete file listing (metadata only, no download)...")
        listing_cache = workdir / ref_to_dirname(ref) / "file_listing.json"
        try:
            files = list_files(api, style, ref, cache_path=listing_cache)
            listing = analyze_listing(files)
        except Exception as exc:
            print(f"[WARN] File listing failed: {exc}")
            files = []
            listing = analyze_listing([])

        if not files:
            print("  [WARN] No file listing available for this dataset")
            results.append({"ref": ref, "error": "file listing unavailable"})
            continue

        print(f"\nTotal files    : {listing['file_count']}")
        print(f"Total size     : {format_size(listing['total_bytes'])}")
        print(f"Image folders  : {listing.get('image_directory_count', 0)}")

        print("\nExtensions:")
        for ext, count in list(listing["extensions"].items())[:10]:
            print(f"  {ext:10s} {count:>7d}")

        # Class / split analysis derived purely from the file listing.
        class_counts = {}
        split_counts = {}
        class_splits = {}
        for record in files:
            name = record.get("name", "") if isinstance(record, dict) else getattr(record, "name", "")
            if not name:
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
                continue
            split, class_name = detect_split_and_class(name)
            if class_name:
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if split:
                split_counts[split] = split_counts.get(split, 0) + 1
                if class_name:
                    class_splits.setdefault(class_name, set()).add(split)

        print(f"\nDetected classes: {len(class_counts)}")
        print(f"{'Class':<52} {'Images':>8}  Splits")
        print("-" * 78)
        for class_name in sorted(class_counts):
            splits = ",".join(sorted(class_splits.get(class_name, set()))) or "-"
            print(f"{class_name:<52} {class_counts[class_name]:>8d}  {splits}")

        if split_counts:
            print("\nSplit totals:")
            for split_name in sorted(split_counts):
                print(f"  {split_name:<12} {split_counts[split_name]:>8d}")

        results.append(
            {
                "ref": ref,
                "title": meta.get("title", ""),
                "owner": meta.get("ownerUser", ""),
                "slug": meta.get("datasetSlug", ""),
                "subtitle": meta.get("subtitle", ""),
                "url": f"https://www.kaggle.com/datasets/{ref}",
                "last_updated": str(meta.get("lastUpdated", "")),
                "licenses": license_names,
                "usability_rating": meta.get("usabilityRating"),
                "votes": meta.get("totalVotes"),
                "views": meta.get("totalViews"),
                "listing": listing,
                "class_counts": class_counts,
                "split_counts": split_counts,
                "detected_splits": sorted(split_counts.keys()),
            }
        )
        print()

    out_dir = workspace_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_metadata.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[OK] Full metadata written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
