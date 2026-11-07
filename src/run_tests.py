# Step 5: score the local methods on the ladder and the labelled test sets, with timing.
# The cutoffs are fixed from validation; nothing is tuned here.

import json
import os
import time

import numpy as np
import pandas as pd

from methods import DATA, RESULTS, build_all, load

thresholds = json.load(open(os.path.join(RESULTS, "thresholds.json")))
scorers = build_all()

ladder = pd.read_csv(os.path.join(DATA, "ladder.csv")).rename(columns={"s1": "sentence1", "s2": "sentence2"})
ladder["dataset"] = "ladder"
qqp = pd.read_csv(os.path.join(RESULTS, "qqp_heldout.csv"))
tests = pd.concat([ladder, load("paws", "test"), load("mrpc", "test"), qqp], ignore_index=True)

timing = {}
for name, scorer in scorers.items():
    a, b = tests["sentence1"].tolist(), tests["sentence2"].tolist()
    start = time.perf_counter()
    scores = scorer.score(a, b)
    seconds = time.perf_counter() - start
    timing[name] = seconds / len(tests) * 1000
    tests[name + "_score"] = scores
    tests[name + "_pred"] = (scores >= thresholds[name]["cutoff"]).astype(int)
    print(name, f"{timing[name]:.2f} s per 1000 pairs", flush=True)

tests.to_csv(os.path.join(RESULTS, "predictions_local.csv"), index=False, encoding="utf-8")
json.dump(timing, open(os.path.join(RESULTS, "timing_local.json"), "w"), indent=2)
print("saved results/predictions_local.csv")
