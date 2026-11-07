# Steps 2-3: apply the meaning review, then build the test set (data/ladder.csv).
# Rejected = meaning changed, information added or lost, a name corrupted, or the sentence is unreadable.
# Unchanged rows are dropped too because they aren't paraphrases. Review was done by Claude, helped by the NLI score.
# Main reasons by level (the review recorded a decision per row, not a reason per row):
#   L1  wrong-sense or awkward synonym ("element" -> "ingredient"), broken verb ("did Luther fright")
#   L2  relationship reversed ("cost Australia victory" -> "Australia's victory was cost"), garbled word order
#   L3  a small word change that shifts the meaning ("exact science" -> "exact field")
#   L4  roles flipped or facts added or lost ("coming weeks" -> "forthcoming days"), question changed
#   L5  corrupted names ("Pine Gap" -> "pine Valley", "the Expos" -> "the fair"), lost information, garbled output

import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

REJECTED = {
    1: {5, 16, 20, 41, 44, 50, 52, 55, 61, 67, 70, 74, 79, 80, 81, 102, 103, 108, 118, 120, 123, 126, 136, 140, 144,
        146, 149, 157, 172, 175, 177, 178, 179, 180, 186, 189, 191, 213, 219, 232, 235, 236, 243, 244, 257, 267,
        269, 271, 274, 296, 298},
    2: {0, 14, 17, 19, 32, 41, 48, 53, 54, 62, 63, 67, 76, 93, 97, 106, 113, 121, 128, 135, 136, 151, 154, 157, 159,
        173, 176, 177, 210, 229, 250, 260, 261, 268, 277},
    3: {31, 136, 239, 256},
    4: {2, 3, 14, 16, 17, 26, 28, 31, 32, 35, 46, 52, 53, 58, 64, 72, 82, 83, 94, 101, 112, 121, 122, 131, 133, 136,
        137, 141, 154, 157, 158, 164, 166, 168, 169, 173, 175, 176, 177, 185, 189, 191, 192, 193, 195, 197, 200, 201,
        204, 208, 211, 214, 223, 225, 232, 234, 236, 237, 238, 250, 251, 260, 261, 264, 267, 268, 269, 272, 273, 276,
        277, 281, 284, 292, 295},
    5: {0, 1, 4, 5, 6, 7, 9, 12, 13, 14, 15, 16, 17, 19, 22, 23, 25, 26, 28, 29, 30, 31, 32, 33, 35, 36, 41, 43, 44, 46,
        47, 48, 51, 52, 53, 54, 58, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 74, 76, 77, 78, 79, 80, 82, 84, 85, 86, 87,
        88, 89, 90, 94, 95, 96, 100, 101, 102, 103, 105, 107, 108, 109, 113, 114, 116, 117, 118, 119, 121, 122, 123,
        124, 126, 127, 129, 131, 132, 135, 136, 141, 146, 147, 149, 150, 151, 152, 153, 154, 155, 158, 159, 160, 161,
        164, 165, 168, 169, 170, 171, 172, 173, 174, 176, 179, 181, 182, 185, 188, 189, 194, 196, 197, 199, 200, 201,
        203, 206, 207, 212, 213, 214, 216, 217, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 234, 237,
        238, 240, 241, 243, 245, 248, 249, 250, 252, 254, 256, 257, 262, 263, 265, 266, 267, 268, 269, 271, 272, 273,
        274, 282, 284, 285, 286, 289, 290, 294, 295, 297, 298, 299},
}

checks = pd.read_csv(os.path.join(DATA, "meaning_check.csv"))
base = pd.read_csv(os.path.join(DATA, "base_sentences.csv"))
checks = checks.merge(base[["id", "source"]], on="id")
checks["rejected"] = [i in REJECTED[l] for i, l in zip(checks["id"], checks["level"])]
checks["keep"] = ~checks["rejected"] & ~checks["unchanged"]
checks.to_csv(os.path.join(DATA, "meaning_review.csv"), index=False, encoding="utf-8")

# Different-meaning pairs: the original vs the same-level paraphrase of the most similar other sentence from the same source
vec = TfidfVectorizer(token_pattern=r"\w+").fit(base["sentence"])
base_vectors = vec.transform(base["sentence"])
similarity = (base_vectors @ base_vectors.T).toarray()
np.fill_diagonal(similarity, -1)
position = {sid: i for i, sid in enumerate(base["id"])}

rows, negative_sims = [], []
for level, group in checks[checks["keep"]].groupby("level"):
    by_id = group.set_index("id")
    for sid, row in by_id.iterrows():
        rows.append(dict(id=sid, level=level, source=row["source"], s1=row["original"], s2=row["paraphrase"], label=1))
        same_source = by_id.index[(by_id["source"] == row["source"]) & (by_id.index != sid)]
        best = max(same_source, key=lambda j: similarity[position[sid], position[j]])
        negative_sims.append(similarity[position[sid], position[best]])
        rows.append(dict(id=sid, level=level, source=row["source"], s1=row["original"],
                         s2=by_id.loc[best, "paraphrase"], label=0))

ladder = pd.DataFrame(rows)
ladder.to_csv(os.path.join(DATA, "ladder.csv"), index=False, encoding="utf-8")

summary = checks.groupby("level").agg(total=("id", "count"), unchanged=("unchanged", "sum"),
                                      rejected=("rejected", "sum"), kept=("keep", "sum"))
kept_ids = [set(g["id"]) for _, g in checks[checks["keep"]].groupby("level")]
print(summary.to_string())
print("kept by source:", checks[checks["keep"]].groupby(["level", "source"]).size().unstack().to_dict("index"))
print("ladder pairs:", len(ladder), "| ids kept at all 5 levels:", len(set.intersection(*kept_ids)))
print("negative-pair source similarity: mean %.2f, max %.2f" % (np.mean(negative_sims), np.max(negative_sims)))
summary.to_csv(os.path.join(DATA, "..", "results", "meaning_summary.csv"))
