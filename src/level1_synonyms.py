# Level 1: swap 2-3 content words for WordNet synonyms
#
# Also writes data/level1_review.csv for hand-checking. Mark the "keep" column
# y (good) or n (meaning changed / broken), then rerun make_base_sentences.py
# and this script - rejected sentences get replaced with fresh ones.

import os
import random
import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.tokenize.treebank import TreebankWordDetokenizer

for pkg in ["punkt_tab", "averaged_perceptron_tagger_eng", "wordnet"]:
    nltk.download(pkg, quiet=True)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
REVIEW = os.path.join(DATA, "level1_review.csv")

QUESTION_WORDS = {"what", "who", "whom", "whose", "which", "when", "where", "why", "how"}
# "is", "did", "have" etc. get tagged as verbs, but swapping them just breaks the sentence.
# Small words like "other" or "next" have terrible WordNet synonyms ("early", "adjacent").
SKIP = {"be", "is", "are", "was", "were", "been", "do", "does", "did", "have", "has", "had",
        "other", "same", "next", "last", "first", "many", "much", "more", "most", "few",
        "several", "such", "own", "only", "new", "old", "get", "make", "let", "take", "go"}

# Known-bad swaps ("time" -> "clip"), kept in a plain text file so it's easy to add to.
# A line with one word means "never swap this word at all".
with open(os.path.join(os.path.dirname(__file__), "bad_swaps.txt")) as f:
    lines = [line.split() for line in f if line.strip() and not line.startswith("#")]
BAD_SWAPS = {tuple(l) for l in lines if len(l) == 2}
BAD_WORDS = {l[0] for l in lines if len(l) == 1}

# Only base-form tags: singular nouns, plain verbs, plain adjectives.
# Plurals, past tenses etc. would need re-inflecting the synonym, so we leave them alone.
POS_MAP = {"NN": wn.NOUN, "VB": wn.VERB, "VBP": wn.VERB, "JJ": wn.ADJ}


def synonym(word, pos):
    if word.lower() in BAD_WORDS:
        return None
    # Only touch words already in their base form, so "finished" doesn't become "end"
    if wn.morphy(word.lower(), pos) != word.lower():
        return None
    # Only the word's most common sense - otherwise "player" turns into "musician".
    # Among its synonyms, pick the one people actually use most (count > 0).
    synsets = wn.synsets(word, pos=pos)
    if not synsets:
        return None
    best = None
    for lemma in synsets[0].lemmas():
        name = lemma.name()
        if "_" in name or name.lower() == word.lower() or lemma.count() == 0:
            continue
        if (word.lower(), name.lower()) in BAD_SWAPS:
            continue
        # Skip abbreviations ("yr") and synonyms in a different form ("vote" -> "voting")
        if len(name) < 3 or wn.morphy(name.lower(), pos) != name.lower():
            continue
        if best is None or lemma.count() > best.count():
            best = lemma
    return best.name() if best else None


def paraphrase(sentence):
    # Seeded by the sentence itself, so the same sentence always gets the same
    # paraphrase no matter which script calls this or in what order
    rng = random.Random(sentence)
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)

    options = []
    for i, (word, tag) in enumerate(tagged):
        pos = POS_MAP.get(tag)
        if pos is None or not word.isalpha() or len(word) < 3:
            continue
        # Capitalised words are usually names or titles ("Doctor Who"). Even the first
        # word can be one ("Murphy" -> "Potato"), so leave all of them alone.
        if word[0].isupper():
            continue
        if word.lower() in QUESTION_WORDS or word.lower() in SKIP:
            continue
        new = synonym(word, pos)
        if new:
            options.append((i, new))

    n = min(len(options), rng.choice([2, 3]))
    for i, new in rng.sample(options, n):
        tokens[i] = new
        # Fix "a agenda" -> "an agenda" and the other way round
        if i > 0 and tokens[i - 1].lower() in ("a", "an"):
            article = "an" if new[0].lower() in "aeiou" else "a"
            tokens[i - 1] = article.capitalize() if tokens[i - 1][0].isupper() else article

    out = TreebankWordDetokenizer().detokenize(tokens)
    # The detokenizer mangles a bit of punctuation - put it back the way it was
    if "&" in sentence and " & " not in sentence:
        out = out.replace(" & ", "&")  # "V & A" -> "V&A"
    if " -- " in sentence:
        out = out.replace("--", " -- ")
    return out, n


if __name__ == "__main__":
    base = pd.read_csv(os.path.join(DATA, "base_sentences.csv"))
    rows = []
    for _, r in base.iterrows():
        new, n = paraphrase(r["sentence"])
        rows.append({"id": r["id"], "source": r["source"], "original": r["sentence"], "paraphrase": new})
    level1 = pd.DataFrame(rows)
    level1[["id", "original", "paraphrase"]].to_csv(os.path.join(DATA, "level1.csv"), index=False)
    print(f"level 1: {len(level1)} sentences -> data/level1.csv")

    # Review file: keep earlier y/n marks, add new rows blank. Marks are matched on
    # original + paraphrase, so if the paraphrase ever changes it gets checked again.
    # Rejected rows stay in the file so they're remembered.
    level1["keep"] = ""
    if os.path.exists(REVIEW):
        old = pd.read_csv(REVIEW, dtype=str, keep_default_na=False)
        old["keep"] = old["keep"].str.strip().str.lower().str[:1]  # "Yes", "Y " -> "y"
        marks = {(o, p): k for o, p, k in zip(old["original"], old["paraphrase"], old["keep"])}
        level1["keep"] = [marks.get((o, p), "") for o, p in zip(level1["original"], level1["paraphrase"])]
        rejected = old[(old["keep"] == "n") & ~old["original"].isin(level1["original"])]
        level1 = pd.concat([level1, rejected.assign(id="")], ignore_index=True)

    level1 = level1.sort_values("keep", key=lambda k: k.map({"": 0, "y": 1, "n": 2}), kind="stable")
    level1.to_csv(REVIEW, index=False)
    todo = (level1["keep"] == "").sum()
    print(f"review: {todo} rows still need a y/n -> data/level1_review.csv")
