# Plan

**Project:** Can computers tell two sentences mean the same, even when the words change?

**Idea:** Rewrite sentences at 5 levels → test the methods → graph who breaks first.

All steps are done. This is what was actually built; `docs/PROGRESS.md` explains why each choice was made, and `README.md` gives the run order.

## Steps

1. **Setup:** folders, Python venv, libraries.
2. **Get data:** PAWS, QQP, MRPC, plus 300 base sentences (150 SQuAD questions, 150 AG News statements).
3. **Make 5 levels** of each sentence:
   - L1: swap words with synonyms (WordNet)
   - L2: change structure (Qwen, "only restructure")
   - L3: minimal LLM rewrite (low temperature)
   - L4: aggressive LLM rewrite (high temperature)
   - L5: translate English → Chinese → English (MarianMT)
   - Meaning check: NLI score plus Claude reading all 1,500 pairs, then a human check of 150 pairs.
4. **Build the methods** (each answers "same meaning? yes/no"):
   - TF-IDF and BM25 (word overlap)
   - SBERT (meaning vectors)
   - Two classifiers: logistic regression on TF-IDF or SBERT vectors, trained on PAWS + QQP + MRPC
   - LLM judge (Gemini)
5. **Run tests**
   - All methods on all 5 levels (the LLM on a sample)
   - All methods on PAWS trick pairs (same words, different meaning)
   - Time and cost of each method
   - Extra checks: cutoffs, acceptance against word overlap
6. **Graphs**
   - Accuracy vs level (main result)
   - Same-meaning pairs accepted vs level and vs word overlap
   - PAWS trick test
   - Cost vs accuracy
7. **Write report:** ACL format, in `report/`.

## Folders
- `data/`: datasets and generated sentences
- `src/`: code
- `results/`: scores and graphs
- `docs/`: this plan and progress
- `report/`: final report
- `presentation/`: mid-term deck
