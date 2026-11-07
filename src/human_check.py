# Human check of the meaning review: a blind sample of 150 pairs, marked by a person.
# Usage: python src/human_check.py make    (writes the sheet and a separate answer key)
#        python src/human_check.py score   (after the sheet is filled in; these are the blind answers)
#        python src/human_check.py review  (blind answers plus any changes in data/human_check_review.csv)

import os
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
SHEET = os.path.join(ROOT, "data", "human_check_sheet.csv")
KEY = os.path.join(ROOT, "data", "human_check_key.csv")
RESULT = os.path.join(ROOT, "results", "human_check.csv")
SEED = 42
PER_LEVEL, REJECTED = 30, 8


def make():
    review = pd.read_csv(os.path.join(ROOT, "data", "meaning_review.csv"))
    review = review[~review["unchanged"]]
    picks = []
    for _, group in review.groupby("level"):
        rejected = group[group["rejected"]]
        rejected = rejected.sample(min(REJECTED, len(rejected)), random_state=SEED)
        kept = group[~group["rejected"]].sample(PER_LEVEL - len(rejected), random_state=SEED)
        picks += [rejected, kept]
    sample = pd.concat(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample["row"] = sample.index + 1
    sheet = sample.rename(columns={"original": "sentence_1", "paraphrase": "sentence_2"})[["row", "sentence_1", "sentence_2"]]
    sheet["same_meaning"] = ""
    sheet.to_csv(SHEET, index=False, encoding="utf-8-sig")
    sample.assign(claude_same=~sample["rejected"])[["row", "id", "level", "source", "claude_same"]].to_csv(KEY, index=False)
    print(f"{len(sheet)} rows; fill in same_meaning with y or n in {SHEET}")


def score():
    sheet = pd.read_csv(SHEET, encoding="utf-8-sig")
    answers = sheet["same_meaning"].astype(str).str.strip().str.lower()
    if not answers.isin(["y", "n"]).all():
        print("rows still blank or not y/n:", sheet.loc[~answers.isin(["y", "n"]), "row"].tolist())
        return
    summary(pd.read_csv(KEY).assign(human_same=(answers == "y").values), RESULT)


def summary(d, path):
    rows = []
    for level, g in list(d.groupby("level")) + [("all", d)]:
        kept, rejected = g[g["claude_same"]], g[~g["claude_same"]]
        rows.append({"level": level, "n": len(g), "agreement": (g["human_same"] == g["claude_same"]).mean(),
                     "kept_n": len(kept), "kept_human_same": kept["human_same"].mean(),
                     "rejected_n": len(rejected), "rejected_human_different": (~rejected["human_same"]).mean()})
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    print(out.round(3).to_string())
    print("Cohen's kappa:", round(cohen_kappa_score(d["human_same"], d["claude_same"]), 3))


def review():
    blind = pd.read_csv(os.path.join(ROOT, "data", "human_check_firstpass.csv"), encoding="utf-8-sig")
    changes = pd.read_csv(os.path.join(ROOT, "data", "human_check_review.csv"), encoding="utf-8-sig")
    answers = blind.set_index("row")["same_meaning"].str.strip().str.lower()
    second = changes.set_index("row")["second_pass"].astype(str).str.strip().str.lower()
    changed = 0
    for row, mark in second[second.isin(["y", "n"])].items():
        changed += answers[row] != mark
        answers[row] = mark
    d = pd.read_csv(KEY).assign(human_same=(answers == "y").values)
    summary(d, os.path.join(ROOT, "results", "human_check_reviewed.csv"))
    print("answers changed:", changed)


{"make": make, "score": score, "review": review}[sys.argv[1]]()
