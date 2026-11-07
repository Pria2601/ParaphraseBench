# Basic checks on the 5 levels: row counts, unchanged rows, word overlap with the original

import os
import re
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
FILES = {1: "level1", 2: "level2", 3: "level3", 4: "level4", 5: "level5_marian_zh"}


def words(s):
    return set(re.findall(r"\w+", str(s).lower()))


def overlap(a, b):
    a, b = words(a), words(b)
    return len(a & b) / len(a | b)


base = pd.read_csv(os.path.join(DATA, "base_sentences.csv"))
print("base", len(base), base["source"].value_counts().to_dict())

for level, name in FILES.items():
    d = pd.read_csv(os.path.join(DATA, name + ".csv")).merge(base[["id", "source"]], on="id")
    d["overlap"] = [overlap(a, b) for a, b in zip(d["original"], d["paraphrase"])]
    same = (d["original"].str.strip() == d["paraphrase"].astype(str).str.strip()).sum()
    by_source = d.groupby("source")["overlap"].mean().round(2).to_dict()
    print(f"L{level}: rows {len(d)}, missing {d['paraphrase'].isna().sum()}, unchanged {same}, "
          f"overlap {d['overlap'].mean():.2f} {by_source}")
