"""Check dataset file listing pagination behaviour."""

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

api = get_kaggle_api()["api"]
REF = "karagwaanntreasure/plant-disease-detection"

total = 0
token = None
pages = 0
while pages < 12:
    resp = api.dataset_list_files(REF, page_token=token, page_size=1000)
    files = list(getattr(resp, "files", None) or [])
    total += len(files)
    pages += 1
    token = getattr(resp, "next_page_token", None)
    print(f"page {pages}: +{len(files)} total={total} next_token={'yes' if token else 'no'}")
    if not token or not files:
        break

print("FINAL TOTAL:", total)

# Inspect a single file record
resp = api.dataset_list_files(REF, page_size=3)
f0 = (getattr(resp, "files", None) or [None])[0]
print()
print("file record type:", type(f0))
print("file record dir:", [a for a in dir(f0) if not a.startswith("_")][:40] if f0 else None)
if f0 is not None:
    print("to_dict:", f0.to_dict())
