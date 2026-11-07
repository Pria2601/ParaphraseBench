# Train the models and pick one cutoff per method on validation data only.
# PAWS, MRPC and QQP count equally, so the larger datasets don't decide the cutoff alone.

import json
import os

import numpy as np
import pandas as pd

from methods import RESULTS, SEED, build_all, load


def best_cutoff(scores, labels, weights):
    order = np.argsort(scores)
    s, y, w = scores[order], labels[order], weights[order]
    # predicted "same" when score >= cutoff; start with everything predicted "same"
    correct = (w * y).sum()
    best, cutoff = correct, s[0] - 1e-9
    for i in range(len(s)):
        correct += w[i] if y[i] == 0 else -w[i]
        if (i + 1 == len(s) or s[i + 1] > s[i]) and correct > best:
            best = correct
            cutoff = (s[i] + s[i + 1]) / 2 if i + 1 < len(s) else s[i] + 1e-9
    return float(cutoff), float(best / w.sum())


qqp = load("qqp", "validation").sample(frac=1, random_state=SEED).reset_index(drop=True)
os.makedirs(RESULTS, exist_ok=True)
qqp.iloc[5000:].to_csv(os.path.join(RESULTS, "qqp_heldout.csv"), index=False)  # QQP has no labelled test split
val = pd.concat([load("paws", "validation"), load("mrpc", "validation"), qqp.iloc[:5000]], ignore_index=True)
weights = 1 / val.groupby("dataset")["label"].transform("count").values

scorers = build_all()
out = {}
a, b, y = val["sentence1"].tolist(), val["sentence2"].tolist(), val["label"].values
for name, scorer in scorers.items():
    scores = scorer.score(a, b)
    cutoff, acc = best_cutoff(scores, y, weights)
    per = {d: float(((scores[val["dataset"] == d] >= cutoff) == y[val["dataset"] == d]).mean())
           for d in ["paws", "mrpc", "qqp"]}
    out[name] = {"cutoff": cutoff, "val_accuracy": acc, "per_dataset": per}
    print(name, round(cutoff, 3), round(acc, 3), {k: round(v, 3) for k, v in per.items()}, flush=True)

json.dump(out, open(os.path.join(RESULTS, "thresholds.json"), "w"), indent=2)
print("saved results/thresholds.json")
