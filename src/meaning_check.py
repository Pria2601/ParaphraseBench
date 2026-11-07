# Step 2: score every original/paraphrase pair with an NLI model, in both directions.
# A pair with low entailment either way is flagged for review. SBERT is not used here because it is one of the tested methods.

import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL = "cross-encoder/nli-MiniLM2-L6-H768"  # small enough to run on a laptop CPU
FILES = {1: "level1", 2: "level2", 3: "level3", 4: "level4", 5: "level5_marian_zh"}

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
ENT = model.config.label2id["entailment"]


def entail(premises, hypotheses, batch=16):
    out = []
    for i in range(0, len(premises), batch):
        enc = tok(premises[i:i + batch], hypotheses[i:i + batch], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out += torch.softmax(model(**enc).logits, dim=-1)[:, ENT].tolist()
    return out


rows = []
for level, name in FILES.items():
    d = pd.read_csv(os.path.join(DATA, name + ".csv"))[["id", "original", "paraphrase"]]
    d["level"] = level
    a, b = d["original"].tolist(), d["paraphrase"].astype(str).tolist()
    d["forward"] = entail(a, b)
    d["backward"] = entail(b, a)
    d["score"] = d[["forward", "backward"]].min(axis=1)
    d["unchanged"] = d["original"].str.strip() == d["paraphrase"].astype(str).str.strip()
    rows.append(d)
    print(f"L{level}: mean {d['score'].mean():.2f}, below 0.5: {(d['score'] < 0.5).sum()}", flush=True)

pd.concat(rows).to_csv(os.path.join(DATA, "meaning_check.csv"), index=False, encoding="utf-8")
print("saved data/meaning_check.csv")
