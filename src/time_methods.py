# Timing for the local methods on the same 1,000 ladder pairs, median of 3 runs.
# run_tests.py's timing can't be used because the laptop slept during that run.

import json
import os
import statistics
import time

import pandas as pd

from methods import DATA, RESULTS, SEED, build_all

scorers = build_all()
pairs = pd.read_csv(os.path.join(DATA, "ladder.csv")).sample(1000, random_state=SEED)
a, b = pairs["s1"].tolist(), pairs["s2"].tolist()

timing = {}
for name, scorer in scorers.items():
    runs = []
    for _ in range(3):
        start = time.perf_counter()
        scorer.score(a, b)
        runs.append(time.perf_counter() - start)
    timing[name] = statistics.median(runs)
    print(name, round(timing[name], 3), flush=True)

json.dump(timing, open(os.path.join(RESULTS, "timing_local.json"), "w"), indent=2)
