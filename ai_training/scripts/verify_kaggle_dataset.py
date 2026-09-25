#!/usr/bin/env python3
"""
Phase 1: Identify and verify Kaggle plant disease datasets.

Uses the Kaggle API. Supports both kaggle 1.x (KaggleApi class) and
kaggle 2.x (module-level functions) so it works across versions.

Credentials are read from environment variables or a local .env file.
The API key is never printed in full.
"""

import io
import os
import sys
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SEARCH_TERMS = [
    "plant disease detection",
    "new plant disease detection",
    "plantvillage",
    "plant disease",
    "leaf disease",
]


def load_env_file(env_path):
    """Load KEY=VALUE pairs from a .env file."""
    env_vars = {}
    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                value = value.strip().strip('"').strip("'")
                env_vars[key.strip()] = value
        return env_vars
    except FileNotFoundError:
        return {}


def resolve_env_file():
    """Locate the .env file, searching upward from this script."""
    script_dir = Path(__file__).resolve().parent
    for candidate_dir in (script_dir.parent, script_dir.parent.parent, Path.cwd()):
        candidate = candidate_dir / ".env"
        if candidate.exists():
            return candidate
    return script_dir.parent / ".env"


def get_kaggle_api():
    """
    Return a Kaggle API handle.

    Returns a dict with 'api' (KaggleApi instance or None) and 'style'
    ('legacy' for kaggle 1.x, 'modern' for kaggle 2.x).
    """
    import kaggle
    import kaggle.api as kaggle_api

    if hasattr(kaggle_api, "authenticate") and not hasattr(kaggle_api, "KaggleApi"):
        # kaggle 2.x: module-level functions
        kaggle_api.authenticate()
        return {"api": kaggle_api, "style": "modern"}

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return {"api": api, "style": "legacy"}


def list_datasets(handle, search_term, page=1):
    """List datasets for a search term across API versions."""
    api, style = handle["api"], handle["style"]
    if style == "modern":
        return api.dataset_list(search=search_term, sort_by="hottest", page=page) or []
    return api.dataset_list(search_term, sort_by="relevance", sort_dir="desc", page=page) or []


def describe_dataset(dataset):
    """Extract comparable metadata fields from a dataset record."""
    def pick(*names, default=None):
        for name in names:
            if isinstance(dataset, dict):
                if name in dataset and dataset[name] is not None:
                    return dataset[name]
            elif hasattr(dataset, name):
                value = getattr(dataset, name)
                if value is not None:
                    return value
        return default

    owner = pick("ownerUsername", "owner", "ownerRef", "ref_owner", default="")
    slug = pick("slug", "datasetSlug", "titleSlug", default="")
    title = pick("title", "name", default="")
    ref = pick("ref", "datasetRef", default="")
    if not ref and owner and slug:
        ref = f"{owner}/{slug}"

    return {
        "title": title,
        "owner": owner,
        "slug": slug,
        "ref": ref,
        "url": f"https://www.kaggle.com/datasets/{ref}" if ref else "",
        "total_bytes": pick("totalBytes", "datasetFilesSize", default=None),
        "file_count": pick("fileCount", "totalFiles", default=None),
        "last_updated": str(pick("lastUpdated", "datasetLastUpdated", default="") or ""),
        "subtitle": str(pick("subtitle", "description", default="") or ""),
        "usability_rating": pick("usabilityRating", default=None),
        "vote_count": pick("totalVotes", "voteCount", default=None),
    }


def format_size(num_bytes):
    """Format a byte count for human display."""
    if not num_bytes:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def is_plant_disease_dataset(info):
    """Heuristic filter for plant leaf disease datasets."""
    haystack = f"{info['title']} {info['subtitle']}".lower()
    has_plant = any(term in haystack for term in ("plant", "leaf", "crop", "tomato", "potato", "apple"))
    has_disease = any(term in haystack for term in ("disease", "blight", "rust", "mildew", "spot", "pest"))
    return has_plant and has_disease


def write_report(workspace_root, candidates, error=None):
    """Write the Phase 1 dataset report."""
    reports_dir = workspace_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "dataset_report.md"
    json_path = reports_dir / "dataset_report.json"

    lines = ["# Phase 1 - Kaggle Dataset Identification", ""]
    payload = {"generated": None, "datasets": candidates, "error": error}

    if error:
        lines += ["## Status", "", f"BLOCKED - {error}", ""]
    else:
        lines += [
            "## Status",
            "",
            f"Authentication: SUCCESS",
            f"Candidate datasets found: {len(candidates)}",
            "",
            "## Candidates",
            "",
        ]
        for idx, info in enumerate(candidates, start=1):
            lines += [
                f"### {idx}. {info['title']}",
                "",
                f"- Owner: {info['owner']}",
                f"- Slug: {info['slug']}",
                f"- Dataset ID: {info['ref']}",
                f"- URL: {info['url']}",
                f"- Size: {format_size(info['total_bytes'])}",
                f"- File count: {info['file_count'] if info['file_count'] is not None else 'unknown'}",
                f"- Last updated: {info['last_updated'] or 'unknown'}",
                "",
            ]

        if candidates:
            selected = candidates[0]
            lines += [
                "## Selected Dataset",
                "",
                f"- Dataset ID: {selected['ref']}",
                f"- URL: {selected['url']}",
                "",
            ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Report written to {report_path}")
    print(f"[OK] Machine-readable report written to {json_path}")


def main():
    workspace_root = Path(__file__).resolve().parent.parent
    env_file = resolve_env_file()

    print("=" * 60)
    print("PHASE 1 - KAGGLE DATASET IDENTIFICATION")
    print("=" * 60)

    env_vars = load_env_file(env_file)
    username = env_vars.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USERNAME")
    key = env_vars.get("KAGGLE_KEY") or os.environ.get("KAGGLE_KEY")

    if not username or not key:
        missing = []
        if not username:
            missing.append("KAGGLE_USERNAME")
        if not key:
            missing.append("KAGGLE_KEY")
        print(f"[FAIL] Missing required configuration: {', '.join(missing)}")
        print(f"       Set these in {env_file} or export them as environment variables.")
        write_report(workspace_root, [], error=f"Missing configuration: {', '.join(missing)}")
        return 1

    # Never print the full key.
    print(f"[OK] KAGGLE_USERNAME: {username}")
    print(f"[OK] KAGGLE_KEY: {key[:4]}{'*' * max(len(key) - 4, 0)} ({len(key)} chars)")
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    try:
        handle = get_kaggle_api()
    except Exception as exc:
        print(f"[FAIL] Kaggle authentication failed: {exc}")
        write_report(workspace_root, [], error=f"Authentication failed: {exc}")
        return 1

    print(f"[OK] Kaggle authentication: SUCCESS (API style: {handle['style']})")

    seen_refs = set()
    candidates = []

    print("\nSearching Kaggle datasets...")
    for term in SEARCH_TERMS:
        try:
            datasets = list_datasets(handle, term)
        except Exception as exc:
            print(f"[WARN] Search for '{term}' failed: {exc}")
            continue

        print(f"  '{term}': {len(datasets)} results")
        for dataset in datasets:
            info = describe_dataset(dataset)
            if not info["ref"] or info["ref"] in seen_refs:
                continue
            seen_refs.add(info["ref"])
            if is_plant_disease_dataset(info):
                candidates.append(info)

    print(f"\n[OK] {len(candidates)} plant-disease dataset candidates identified")

    for idx, info in enumerate(candidates[:10], start=1):
        size = format_size(info["total_bytes"])
        files = info["file_count"] if info["file_count"] is not None else "?"
        print(f"  {idx:2d}. {info['title']}")
        print(f"      id={info['ref']}  size={size}  files={files}")

    write_report(workspace_root, candidates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
