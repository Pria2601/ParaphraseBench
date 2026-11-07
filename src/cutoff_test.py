# How much of each method's drop comes from its cutoff? Scores stay the same, only the cutoff changes:
#   original   tuned on PAWS + MRPC + QQP validation (the main results)
#   no_paws    tuned on MRPC + QQP validation only
#   best       best possible cutoff per ladder level, chosen on the answers, so an upper limit and not a fair score

import json
import os

import numpy as np
import pandas as pd

from methods import RESULTS, SEED, build_all, load

METHODS = ["tfidf", "bm25", "sbert", "clf_tfidf", "clf_sbert"]


def best_cutoff(scores, labels, weights):
    order = np.argsort(scores)
    s, y, w = scores[order], labels[order], weights[order]
    correct = (w * y).sum()
    best, cutoff = correct, s[0] - 1e-9
    for i in range(len(s)):
        correct += w[i] if y[i] == 0 else -w[i]
        if (i + 1 == len(s) or s[i + 1] > s[i]) and correct > best:
            best = correct
            cutoff = (s[i] + s[i + 1]) / 2 if i + 1 < len(s) else s[i] + 1e-9
    return float(cutoff)


# same validation data as tune_thresholds.py, without PAWS
qqp = load("qqp", "validation").sample(frac=1, random_state=SEED).reset_index(drop=True)
val = pd.concat([load("mrpc", "validation"), qqp.iloc[:5000]], ignore_index=True)
weights = 1 / val.groupby("dataset")["label"].transform("count").values
scorers = build_all()
a, b, y = val["sentence1"].tolist(), val["sentence2"].tolist(), val["label"].values
original = json.load(open(os.path.join(RESULTS, "thresholds.json")))
no_paws = {m: best_cutoff(scorers[m].score(a, b), y, weights) for m in METHODS}

preds = pd.read_csv(os.path.join(RESULTS, "predictions_local.csv"), low_memory=False)
ladder = preds[preds["dataset"] == "ladder"].copy()
ladder["level"] = ladder["level"].astype(int)
paws = preds[preds["dataset"] == "paws"]


def accuracy(d, m, cutoff):
    return ((d[m + "_score"] >= cutoff).astype(int) == d["label"]).mean()


rows = []
for m in METHODS:
    row = {"method": m, "cutoff_original": original[m]["cutoff"], "cutoff_no_paws": no_paws[m]}
    for level, d in ladder.groupby("level"):
        row[f"L{level}_original"] = accuracy(d, m, original[m]["cutoff"])
        row[f"L{level}_no_paws"] = accuracy(d, m, no_paws[m])
        scores, labels = d[m + "_score"].values, d["label"].values
        row[f"L{level}_best"] = accuracy(d, m, best_cutoff(scores, labels, np.ones(len(d))))
    row["paws_original"] = accuracy(paws, m, original[m]["cutoff"])
    row["paws_no_paws"] = accuracy(paws, m, no_paws[m])
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RESULTS, "cutoff_test.csv"), index=False)
print(out.round(3).T.to_string())
