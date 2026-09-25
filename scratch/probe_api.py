"""Probe the kaggle 2.x response object shapes."""

import io
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path("ai_training/scripts").resolve()))
from verify_kaggle_dataset import load_env_file, resolve_env_file, get_kaggle_api  # noqa: E402

env_vars = load_env_file(resolve_env_file())
os.environ["KAGGLE_USERNAME"] = env_vars["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = env_vars["KAGGLE_KEY"]

handle = get_kaggle_api()
api = handle["api"]
REF = "karagwaanntreasure/plant-disease-detection"

cache = Path("ai_training/data/metadata_cache/probe")
cache.mkdir(parents=True, exist_ok=True)

print("### dataset_metadata type ###")
meta = api.dataset_metadata(REF, str(cache))
print("type:", type(meta))
print("dir:", [a for a in dir(meta) if not a.startswith("_")][:60])

print()
print("### files written by dataset_metadata ###")
for p in cache.rglob("*"):
    if p.is_file():
        print(" ", p, p.stat().st_size, "bytes")

print()
print("### dataset_list_files ###")
resp = api.dataset_list_files(REF, page_size=20)
print("type:", type(resp))
print("dir:", [a for a in dir(resp) if not a.startswith("_")][:60])
