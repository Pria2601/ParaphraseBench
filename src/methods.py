# Similarity methods: TF-IDF, BM25, SBERT and the trained classifiers.
# Each scorer takes two lists of sentences and returns one score per pair (higher = more similar).

import os
import math
import re
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
SBERT_NAME = "all-MiniLM-L6-v2"
SEED = 42


def load(name, split):
    d = pd.read_csv(os.path.join(DATA, f"{name}_{split}.csv"))
    d = d.rename(columns={"question1": "sentence1", "question2": "sentence2"})
    d = d.dropna(subset=["sentence1", "sentence2"])
    d = d[["sentence1", "sentence2", "label"]].astype({"sentence1": str, "sentence2": str})
    d["dataset"] = name
    return d.reset_index(drop=True)


def train_data():
    qqp = load("qqp", "train").sample(50000, random_state=SEED)
    return pd.concat([load("paws", "train"), load("mrpc", "train"), qqp], ignore_index=True)


def tokens(s):
    return re.findall(r"\w+", s.lower())


class TfidfScorer:
    def __init__(self):
        self.vec = TfidfVectorizer(lowercase=True, token_pattern=r"\w+")

    def fit(self, sentences):
        self.vec.fit(sentences)
        return self

    def vectors(self, s):
        return self.vec.transform(s)

    def score(self, a, b):
        return np.asarray(self.vectors(a).multiply(self.vectors(b)).sum(axis=1)).ravel()


class BM25Scorer:
    # BM25 is one-directional and unbounded, so each direction is divided by the
    # sentence's score against itself, then the two directions are averaged.
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def fit(self, sentences):
        df, total = Counter(), 0
        for s in sentences:
            t = tokens(s)
            total += len(t)
            df.update(set(t))
        n = len(sentences)
        self.avgdl = total / n
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.default_idf = math.log(1 + (n + 0.5) / 0.5)
        return self

    def _one(self, query, doc):
        counts = Counter(doc)
        norm = self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)
        total = 0.0
        for w in query:
            f = counts.get(w, 0)
            if f:
                total += self.idf.get(w, self.default_idf) * f * (self.k1 + 1) / (f + norm)
        return total

    def _directed(self, a, b):
        self_score = self._one(a, a)
        return self._one(a, b) / self_score if self_score else 0.0

    def score(self, a, b):
        out = []
        for x, y in zip(a, b):
            tx, ty = tokens(x), tokens(y)
            out.append((self._directed(tx, ty) + self._directed(ty, tx)) / 2)
        return np.array(out)


class SbertScorer:
    def __init__(self):
        self.model = SentenceTransformer(SBERT_NAME, device="cpu")

    def vectors(self, s):
        return self.model.encode(list(s), batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    def score(self, a, b):
        return (self.vectors(a) * self.vectors(b)).sum(axis=1)


def pair_features(u, v):
    if sparse.issparse(u):
        return sparse.hstack([abs(u - v), u.multiply(v)]).tocsr()
    return np.hstack([np.abs(u - v), u * v])


class ClassifierScorer:
    # Logistic regression on [|u-v|, u*v], so only the sentence representation differs between the two versions.
    def __init__(self, encoder):
        self.encoder = encoder
        self.clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)

    def fit(self, a, b, y):
        self.clf.fit(pair_features(self.encoder.vectors(a), self.encoder.vectors(b)), y)
        return self

    def score(self, a, b):
        return self.clf.predict_proba(pair_features(self.encoder.vectors(a), self.encoder.vectors(b)))[:, 1]


def build_all(cache=True):
    """Fit everything on the training data; reuse saved models when they exist."""
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "models.joblib")
    sbert = SbertScorer()
    if cache and os.path.exists(path):
        saved = joblib.load(path)
        tfidf, bm25, clf_tfidf_model, clf_sbert_model = saved
    else:
        train = train_data()
        a, b, y = train["sentence1"].tolist(), train["sentence2"].tolist(), train["label"].values
        corpus = list(dict.fromkeys(a + b))
        print(f"training on {len(train)} pairs, {len(corpus)} unique sentences", flush=True)
        tfidf = TfidfScorer().fit(corpus)
        bm25 = BM25Scorer().fit(corpus)
        clf_tfidf_model = ClassifierScorer(tfidf).fit(a, b, y).clf
        print("trained TF-IDF classifier", flush=True)
        clf_sbert_model = ClassifierScorer(sbert).fit(a, b, y).clf
        print("trained SBERT classifier", flush=True)
        joblib.dump((tfidf, bm25, clf_tfidf_model, clf_sbert_model), path)
    clf_tfidf = ClassifierScorer(tfidf)
    clf_tfidf.clf = clf_tfidf_model
    clf_sbert = ClassifierScorer(sbert)
    clf_sbert.clf = clf_sbert_model
    return {"tfidf": tfidf, "bm25": bm25, "sbert": sbert, "clf_tfidf": clf_tfidf, "clf_sbert": clf_sbert}
