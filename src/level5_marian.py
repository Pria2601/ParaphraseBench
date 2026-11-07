# Level 5 with MarianMT (Helsinki-NLP opus-mt): try 3 back-translation chains
# Usage: python src/level5_marian.py [N] [chain]   (N = sentences per source; chain = da / zh / da_zh, default all)

import os
import sys
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

# Marian has no Danish->Chinese model, so the third chain goes back through English
CHAINS = {
    "da": ["en", "da", "en"],
    "zh": ["en", "zh", "en"],
    "da_zh": ["en", "da", "en", "zh", "en"],
}

models = {}


def translate(texts, src, tgt, batch=8):
    name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    if name not in models:
        models[name] = (MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name))
    tok, model = models[name]
    if tgt == "zh":
        texts = [">>cmn_Hans<< " + t for t in texts]  # en-zh model needs a target language tag
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True)
        out += tok.batch_decode(model.generate(**enc, max_new_tokens=100), skip_special_tokens=True)
    print(f"  {src}->{tgt}: done", flush=True)
    return out


base = pd.read_csv(os.path.join(DATA, "base_sentences.csv"))
if len(sys.argv) > 1:
    base = base.groupby("source").head(int(sys.argv[1]))  # first N of each source

if len(sys.argv) > 2:
    CHAINS = {sys.argv[2]: CHAINS[sys.argv[2]]}  # run only one chain, e.g. "zh"

for name, chain in CHAINS.items():
    texts = base["sentence"].tolist()
    for src, tgt in zip(chain, chain[1:]):
        texts = translate(texts, src, tgt)
    out = f"level5_marian_{name}.csv"
    pd.DataFrame({"id": base["id"], "original": base["sentence"], "paraphrase": texts}).to_csv(
        os.path.join(DATA, out), index=False, encoding="utf-8")
    print(f"saved data/{out}")
