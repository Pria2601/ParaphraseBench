# Level 5: back-translation through other languages and back to English (NLLB-200)
# Usage: python src/level5_backtranslate.py [N]   (N = sentences per source, default all)

import os
import sys
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL = "facebook/nllb-200-distilled-600M"
CHAIN = ["eng_Latn", "dan_Latn", "zho_Hans", "eng_Latn"]  # English -> Danish -> Chinese -> English
OUT = "level5_da_zh.csv"  # earlier try (Chinese -> Bengali) is in level5.csv

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)


def translate(texts, src, tgt, batch=8):
    tok.src_lang = src
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True)
        gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt), max_new_tokens=100)
        out += tok.batch_decode(gen, skip_special_tokens=True)
        print(f"  {src}->{tgt}: {len(out)}/{len(texts)}", flush=True)
    return out


base = pd.read_csv(os.path.join(DATA, "base_sentences.csv"))
if len(sys.argv) > 1:
    base = base.groupby("source").head(int(sys.argv[1]))  # first N of each source

df = pd.DataFrame({"id": base["id"], "original": base["sentence"]})
texts = base["sentence"].tolist()
for src, tgt in zip(CHAIN, CHAIN[1:]):
    texts = translate(texts, src, tgt)
    df[tgt] = texts  # keep each step for checking

df = df.rename(columns={"eng_Latn": "paraphrase"})
df.to_csv(os.path.join(DATA, OUT), index=False, encoding="utf-8")
print(f"saved {len(df)} rows -> data/{OUT}")
