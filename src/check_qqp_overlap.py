# How many held-out QQP pairs share a question with the training sample?

import json
import os

import pandas as pd

from methods import RESULTS, train_data

train = train_data()
train = train[train["dataset"] == "qqp"]
seen = set(train["sentence1"]) | set(train["sentence2"])
held = pd.read_csv(os.path.join(RESULTS, "qqp_heldout.csv"))
share = float((held["sentence1"].isin(seen) | held["sentence2"].isin(seen)).mean())
json.dump({"share": share, "n": len(held)}, open(os.path.join(RESULTS, "qqp_overlap.json"), "w"), indent=2)
print(round(share, 4), len(held))
