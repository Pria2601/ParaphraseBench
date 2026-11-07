# Step 2b: base sentences for the 5-level ladder
# 150 SQuAD questions + 150 AG News statements, 8-25 words each.
# Only sentences where Level 1 can swap at least 2 words are kept, and anything
# rejected during the hand-check (keep = n in level1_review.csv) is skipped.

import os
import re
import pandas as pd
from datasets import load_dataset
from level1_synonyms import paraphrase, REVIEW

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
N_EACH = 150
SEED = 42


def word_count_ok(s):
    return s.str.split().str.len().between(8, 25)


rejected = set()
if os.path.exists(REVIEW):
    review = pd.read_csv(REVIEW, dtype=str, keep_default_na=False)
    rejected = set(review.loc[review["keep"].str.strip().str.lower().str[:1] == "n", "original"])


def shuffled(s):
    # Fixed seed -> same order -> same picks every run
    return s.sample(frac=1, random_state=SEED)


def pick(candidates, name):
    # Walk down the (already shuffled) list taking good ones
    picked = []
    for s in candidates:
        if s in rejected:
            continue
        if paraphrase(s)[1] >= 2:
            picked.append(s)
        if len(picked) == N_EACH:
            break
    print(f"{name}: picked {len(picked)}")
    return picked


# 1. SQuAD questions (proper ones, ending in "?")
squad = load_dataset("rajpurkar/squad", split="validation").to_pandas()
q = squad["question"].str.strip().drop_duplicates()
q = q[word_count_ok(q) & q.str.endswith("?")]

# 2. AG News statements
# HF text = title + " " + description with no separator, so the first sentence
# is polluted by the title -> only use the 2nd+ sentences of each article.
def ag_sentences(split):
    ag = load_dataset("fancyzhx/ag_news", split=split).to_pandas()
    sents = []
    for text in ag["text"]:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        sents.extend(p.strip() for p in parts[1:])
    s = pd.Series(sents).drop_duplicates()
    s = s[word_count_ok(s)]
    s = s[s.str.match(r"^[A-Z][^\\<>#&;|]*[^.]\.$")]  # clean prose: no HTML/escape junk, ends with "." (not "...")
    s = s[~s.str.startswith("By ") & ~s.str.contains("  ")]  # bylines like "By John Smith  Staff Writer"
    return shuffled(s)


# Test split runs out once enough get rejected, so the train split is the backup.
# It goes after the test split, so earlier picks don't move.
s = pd.concat([ag_sentences("test"), ag_sentences("train")]).drop_duplicates()

base = pd.concat([
    pd.DataFrame({"sentence": pick(shuffled(q), "squad"), "source": "squad"}),
    pd.DataFrame({"sentence": pick(s, "agnews"), "source": "agnews"}),
], ignore_index=True)
base.to_csv(os.path.join(DATA, "base_sentences.csv"), index_label="id")
print(f"base sentences: {len(base)} -> data/base_sentences.csv")
