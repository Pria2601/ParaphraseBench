# Step 2: download datasets and save as CSV in data/

import os
from datasets import load_dataset

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA, exist_ok=True)

# 1. Labelled paraphrase datasets (for training/testing the classifier + PAWS trick test)
sources = {
    "paws": ("google-research-datasets/paws", "labeled_final"),
    "qqp": ("nyu-mll/glue", "qqp"),
    "mrpc": ("nyu-mll/glue", "mrpc"),
}

for name, (path, subset) in sources.items():
    ds = load_dataset(path, subset)
    for split in ds:
        df = ds[split].to_pandas()
        if (df["label"] == -1).all():  # QQP test has hidden labels -> useless
            continue
        out = os.path.join(DATA, f"{name}_{split}.csv")
        df.to_csv(out, index=False)
        print(f"{name} {split}: {len(df)} rows -> {out}")

# Base sentences for the 5-level ladder: see make_base_sentences.py
