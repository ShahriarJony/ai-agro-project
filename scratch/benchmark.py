"""Quick throughput benchmark before committing to a full training run."""

import io
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path("ai_training/scripts").resolve()))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from train_model import LeafDataset, build_transforms, build_model, load_split, WORKSPACE_ROOT

torch.set_num_threads(16)

rows = load_split(WORKSPACE_ROOT / "data" / "manifests" / "dataset_split.csv", "train")
print(f"train rows: {len(rows)}")

subset = rows[:640]
for workers in (8, 12):
    loader = DataLoader(
        LeafDataset(subset, build_transforms(True)),
        batch_size=32,
        shuffle=True,
        num_workers=workers,
        drop_last=True,
    )
    model = build_model(23, pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # warmup
    for images, _ in loader:
        optimizer.zero_grad()
        criterion(model(images), torch.zeros(32, dtype=torch.long))
        break

    start = time.time()
    batches = 0
    for images, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), torch.zeros(32, dtype=torch.long))
        loss.backward()
        optimizer.step()
        batches += 1
    elapsed = time.time() - start

    imgs = batches * 32
    rate = imgs / elapsed
    epoch_minutes = (len(rows) / rate) / 60
    print(
        f"workers={workers:2d}  {elapsed:6.1f}s for {imgs} imgs  "
        f"({rate:6.1f} img/s)  -> full epoch ~{epoch_minutes:5.1f} min"
    )
