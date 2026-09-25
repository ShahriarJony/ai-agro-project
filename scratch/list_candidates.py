import json

with open("ai_training/reports/dataset_report.json", encoding="utf-8") as handle:
    datasets = json.load(handle)["datasets"]

print("=== refs containing 'new' AND 'plant' AND 'disease' ===")
for item in datasets:
    title = item["title"].lower()
    if "new" in title and "plant" in title and "disease" in title:
        print(" ", item["ref"], "|", item["title"], "|", item["subtitle"][:70])

print()
print("=== plantvillage-style refs ===")
for item in datasets:
    ref = item["ref"].lower()
    if "plantvillage" in ref or "plantdisease" in ref or "plant-disease" in ref:
        print(" ", item["ref"], "|", item["title"], "|", item["subtitle"][:70])

print()
print("=== top 20 by title, with ref ===")
for idx, item in enumerate(datasets[:20], start=1):
    print(f" {idx:2d}. {item['ref']:60s} | {item['title'][:45]}")
