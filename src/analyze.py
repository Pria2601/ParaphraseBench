# Step 6: turn predictions into tables and graphs (results/).

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
SEED = 42
LOCAL = ["tfidf", "bm25", "sbert", "clf_tfidf", "clf_sbert"]
NAMES = {"tfidf": "TF-IDF", "bm25": "BM25", "sbert": "SBERT", "clf_tfidf": "Classifier (TF-IDF)",
         "clf_sbert": "Classifier (SBERT)", "gemini": "LLM judge (Gemini, sample)"}
COLORS = {"tfidf": "#7A8794", "bm25": "#B0793B", "sbert": "#2E5BBA", "clf_tfidf": "#5FA777",
          "clf_sbert": "#7A5BB5", "gemini": "#C8443C"}
METHODS = LOCAL + ["gemini"]

preds = pd.read_csv(os.path.join(RESULTS, "predictions_local.csv"))
timing = json.load(open(os.path.join(RESULTS, "timing_local.json")))

replies = {}
gemini_seconds, gemini_tokens, invalid = [], [], 0
with open(os.path.join(RESULTS, "judge_gemini31.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        word = r["reply"].strip().lower()
        if word.startswith("yes"):
            replies[r["key"]] = 1
        elif word.startswith("no"):
            replies[r["key"]] = 0
        else:
            invalid += 1
        gemini_seconds.append(r["seconds"])
        gemini_tokens.append(r["total_tokens"])
preds["gemini_pred"] = (preds["sentence1"] + "\t" + preds["sentence2"]).map(replies)
preds["gemini_score"] = preds["gemini_pred"]


def metrics(d, m):
    d = d[d[m + "_pred"].notna()]
    if d.empty:
        return None
    y, p = d["label"].values, d[m + "_pred"].values.astype(int)
    tp, tn = ((p == 1) & (y == 1)).sum(), ((p == 0) & (y == 0)).sum()
    out = {"n": len(d), "accuracy": (p == y).mean(), "recall_same": tp / max((y == 1).sum(), 1),
           "recall_different": tn / max((y == 0).sum(), 1)}
    if m != "gemini" and len(set(y)) == 2:
        out["auc"] = roc_auc_score(y, d[m + "_score"])
    return out


def ci(d, m, reps=1000):
    # resample base sentences, not single pairs, because the two pairs of a sentence are related
    rng = np.random.default_rng(SEED)
    d = d[d[m + "_pred"].notna()]
    groups = [g for _, g in d.groupby("id")]
    correct = np.array([(g[m + "_pred"].astype(int) == g["label"]).sum() for g in groups])
    total = np.array([len(g) for g in groups])
    picks = rng.integers(0, len(groups), (reps, len(groups)))
    acc = correct[picks].sum(1) / total[picks].sum(1)
    return np.percentile(acc, [2.5, 97.5])


ladder = preds[preds["dataset"] == "ladder"].copy()
ladder["level"] = ladder["level"].astype(int)
rows = []
for level, d in ladder.groupby("level"):
    for m in METHODS:
        s = metrics(d, m)
        if s:
            low, high = ci(d, m)
            rows.append({"level": level, "method": m, "ci_low": low, "ci_high": high, **s})
by_level = pd.DataFrame(rows)
by_level.to_csv(os.path.join(RESULTS, "ladder_by_level.csv"), index=False)

# every method on exactly the pairs the LLM judge answered
sampled = ladder[ladder["gemini_pred"].notna()]
rows = []
for level, d in sampled.groupby("level"):
    for m in METHODS:
        low, high = ci(d, m)
        rows.append({"level": level, "method": m, "ci_low": low, "ci_high": high, **metrics(d, m)})
same_pairs = pd.DataFrame(rows)
same_pairs.to_csv(os.path.join(RESULTS, "ladder_same_pairs.csv"), index=False)

# cross-check: the other Gemini model, on whichever pairs it answered before its daily limit
cross = {}
with open(os.path.join(RESULTS, "judge_gemini.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        word = r["reply"].strip().lower()
        if word.startswith(("yes", "no")):
            cross[r["key"]] = int(word.startswith("yes"))
check = ladder.assign(other=(ladder["sentence1"] + "\t" + ladder["sentence2"]).map(cross)).dropna(subset=["other"])
check = check.groupby("level").apply(lambda d: pd.Series({"n": len(d), "accuracy": (d["other"] == d["label"]).mean()}))
check.to_csv(os.path.join(RESULTS, "gemini_crosscheck.csv"))
both = sampled.assign(other=(sampled["sentence1"] + "\t" + sampled["sentence2"]).map(cross)).dropna(subset=["other"])
agreement = float((both["other"] == both["gemini_pred"]).mean()) if len(both) else None

# same check on the sentences kept at all 5 levels, so level differences can't come from different sentences
pos = ladder[ladder["label"] == 1].groupby("id")["level"].nunique()
common = pos[pos == 5].index
rows = []
for level, d in ladder[ladder["id"].isin(common)].groupby("level"):
    for m in LOCAL:
        rows.append({"level": level, "method": m, **metrics(d, m)})
pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "ladder_common_sentences.csv"), index=False)

# overlap, per level, for same-meaning pairs
def overlap(a, b):
    import re
    a, b = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(a & b) / len(a | b)


pairs = ladder[ladder["label"] == 1]
overlaps = pairs.assign(o=[overlap(a, b) for a, b in zip(pairs["sentence1"], pairs["sentence2"])]).groupby("level")["o"].mean()
overlaps.to_csv(os.path.join(RESULTS, "overlap_by_level.csv"))

# how often same-meaning pairs are accepted at each amount of word overlap, whatever their level
binned = pairs.assign(o=[overlap(a, b) for a, b in zip(pairs["sentence1"], pairs["sentence2"])])
binned["low"] = np.minimum((binned["o"] * 10).astype(int), 9) / 10
acceptance = binned.groupby("low").agg(n=("o", "size"), **{m: (m + "_pred", "mean") for m in LOCAL})
acceptance.to_csv(os.path.join(RESULTS, "acceptance_by_overlap.csv"))

# labelled test sets
paws_full = preds[preds["dataset"] == "paws"]
sample = pd.read_csv(os.path.join(DATA, "paws_test.csv")).groupby("label").sample(250, random_state=SEED)
sample_keys = set(sample["sentence1"] + "\t" + sample["sentence2"])
paws_keys = paws_full["sentence1"] + "\t" + paws_full["sentence2"]
paws_500 = paws_full[paws_keys.isin(sample_keys)].drop_duplicates(["sentence1", "sentence2"])
if paws_500["gemini_pred"].notna().any():  # compare every method on the pairs Gemini actually answered
    paws_500 = paws_500[paws_500["gemini_pred"].notna()]
rows = []
for name, d in [("paws", paws_full), ("paws_500", paws_500), ("mrpc", preds[preds["dataset"] == "mrpc"]),
                ("qqp", preds[preds["dataset"] == "qqp"])]:
    for m in METHODS:
        if m == "gemini" and name != "paws_500":
            continue
        s = metrics(d, m)
        if s:
            rows.append({"dataset": name, "method": m, **s})
tests = pd.DataFrame(rows)

# the same PAWS pairs went to both methods, so compare them with an exact McNemar test
gemini_right = paws_500["gemini_pred"].astype(int) == paws_500["label"]
sbert_right = paws_500["sbert_pred"].astype(int) == paws_500["label"]
only_gemini, only_sbert = int((gemini_right & ~sbert_right).sum()), int((~gemini_right & sbert_right).sum())
mcnemar_p = binomtest(only_gemini, only_gemini + only_sbert, 0.5).pvalue
tests.to_csv(os.path.join(RESULTS, "test_sets.csv"), index=False)

# cost
# median, because free-tier throttling makes a few calls take minutes
gemini_per_1000 = np.median(gemini_seconds) * 1000 if gemini_seconds else float("nan")
cost = pd.DataFrame([{"method": m, "seconds_per_1000": timing[m]} for m in LOCAL] +
                    [{"method": "gemini", "seconds_per_1000": gemini_per_1000}])
cost["ladder_accuracy"] = [metrics(sampled, m)["accuracy"] for m in METHODS]
cost.to_csv(os.path.join(RESULTS, "cost.csv"), index=False)
json.dump({"gemini_calls": len(gemini_seconds), "invalid_replies": invalid,
           "mean_latency_s": float(np.mean(gemini_seconds)) if gemini_seconds else None,
           "median_latency_s": float(np.median(gemini_seconds)) if gemini_seconds else None,
           "mean_tokens": float(np.mean(gemini_tokens)) if gemini_tokens else None,
           "sentences_kept_all_levels": len(common), "model_agreement": agreement, "agreement_n": len(both),
           "mcnemar_only_gemini": only_gemini, "mcnemar_only_sbert": only_sbert, "mcnemar_p": mcnemar_p},
          open(os.path.join(RESULTS, "run_info.json"), "w"), indent=2)

# examples for the failure analysis
examples = []
for level, d in ladder.groupby("level"):
    for m in METHODS:
        wrong = d[d[m + "_pred"].notna() & (d[m + "_pred"] != d["label"])]
        for _, r in wrong.sample(min(3, len(wrong)), random_state=SEED).iterrows():
            examples.append({"level": level, "method": m, "label": r["label"], "s1": r["sentence1"], "s2": r["sentence2"]})
pd.DataFrame(examples).to_csv(os.path.join(RESULTS, "failure_examples.csv"), index=False, encoding="utf-8")

# graphs
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})
for metric, ylabel, name in [("accuracy", "Accuracy", "fig_degradation.png"), ("recall_same", "Same-meaning pairs correct", "fig_recall_same.png")]:
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for m in METHODS:
        d = by_level[by_level["method"] == m].sort_values("level")
        ax.plot(d["level"], d[metric], marker="o", ms=3, color=COLORS[m], label=NAMES[m], lw=1.2)
        if metric == "accuracy":
            ax.fill_between(d["level"], d["ci_low"], d["ci_high"], color=COLORS[m], alpha=0.10)
    ax.axhline(0.5, color="#999999", ls="--", lw=1)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f"L{l}\n({overlaps[l]:.2f})" for l in range(1, 6)])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Paraphrase level (mean word overlap)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=6, ncol=1, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, name), dpi=200)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.5))
d = tests[(tests["dataset"] == "paws_500")].set_index("method").reindex(METHODS).dropna(subset=["accuracy"])
x = np.arange(len(d))
ax.bar(x - 0.2, d["accuracy"], 0.4, color="#1F3A5F", label=f"All {len(paws_500)} pairs")
ax.bar(x + 0.2, d["recall_different"], 0.4, color="#7FA3D9", label="Different-meaning pairs only")
ax.axhline(0.5, color="#999999", ls="--", lw=1)
ax.set_xticks(x)
ax.set_xticklabels([NAMES[m].replace(" (", "\n(").replace(", sample", "").replace("Classifier", "Clf.") for m in d.index], fontsize=6)
ax.set_ylim(0, 1)
ax.set_ylabel("Share correct")
ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig_paws.png"), dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.5))
for _, r in cost.iterrows():
    ax.scatter(r["seconds_per_1000"], r["ladder_accuracy"], s=22, color=COLORS[r["method"]])
    below = r["method"] in ("tfidf", "clf_sbert")  # these sit almost on top of BM25 and SBERT
    last = r["method"] == "gemini"
    ax.annotate(NAMES[r["method"]].replace(", sample", ""), (r["seconds_per_1000"], r["ladder_accuracy"]),
                xytext=(-5, -3) if last else (5, -9 if below else 4), ha="right" if last else "left",
                textcoords="offset points", fontsize=6.5)
ax.set_xscale("log")
ax.set_xlabel("Seconds per 1,000 pairs (log scale)")
ax.set_ylabel("Ladder accuracy")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig_cost.png"), dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.5))
for m in LOCAL:
    ax.plot(acceptance.index + 0.05, acceptance[m], marker="o", ms=3, color=COLORS[m], label=NAMES[m], lw=1.2)
ax.axhline(0.5, color="#999999", ls="--", lw=1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Word overlap with the original (Jaccard)")
ax.set_ylabel("Same-meaning pairs accepted")
ax.legend(frameon=False, fontsize=6, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig_overlap.png"), dpi=200)
plt.close(fig)

print(by_level.pivot(index="level", columns="method", values="accuracy")[METHODS].round(3).to_string())
print(by_level.pivot(index="level", columns="method", values="auc").reindex(columns=LOCAL).round(3).to_string())
print(by_level.pivot(index="level", columns="method", values="n")[METHODS].to_string())
print(same_pairs.pivot(index="level", columns="method", values="accuracy")[METHODS].round(3).to_string())
print(same_pairs[same_pairs["method"] == "gemini"].round(3).to_string())
print(check.round(3).to_string())
print(tests.round(3).to_string())
print(cost.round(3).to_string())
print("run info:", open(os.path.join(RESULTS, "run_info.json")).read())
