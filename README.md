# Meaning-Preserving Paraphrase Robustness

DA6024 course project. Can a computer tell that two sentences mean the same when the words change? We rewrite 300 sentences at five paraphrase levels and test TF-IDF, BM25, SBERT, two trained classifiers and an LLM judge on them, on the PAWS trick test, and on cost.

- **Report:** `report/DA6024_Final_Report_ACL.pdf` (ACL 2023 format; source `report/DA6024_Final_Report_ACL.tex`)
- **Walkthrough notebook:** `DA6024_Execution.ipynb` runs the analysis and shows every table and figure

## What is in the repository

| Folder | Contents |
|---|---|
| `src/` | All code, one script per step |
| `data/` | Our own data: the 300 base sentences, all five levels, the meaning review, the test pairs (`ladder.csv`) and the human-check sheets. The public datasets are not included; `src/download_data.py` fetches them. |
| `results/` | Every output: predictions, tables (CSV and JSON), figures (PNG), LLM replies (JSONL) and the trained models |
| `report/` | Final report and its LaTeX source |

## Setup

Tested with Python 3.12 on Windows. A GPU is not needed.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

On macOS or Linux, use `.venv/bin/python` instead of `.venv/Scripts/python`.

The first run downloads the public datasets and the models (SBERT, the NLI model and MarianMT) from Hugging Face, so it needs internet. After that, set `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` so later runs don't wait on the Hub.

Only two steps call paid or rate-limited APIs: Levels 2 to 4 (Groq) and the LLM judge (Gemini). To rerun them, create a file called `.env` in this folder:

```
GROQ_API_KEY=your-groq-key
GEMINI_API_KEY=your-gemini-key
```

Never upload `.env`; `.gitignore` already excludes it. **You don't need any keys to check our results**: every LLM output is already saved in `data/` and `results/`.

## Quick check (no API keys, a few minutes)

This recreates every table and figure from the saved outputs:

```bash
.venv/Scripts/python src/download_data.py
.venv/Scripts/python src/analyze.py
.venv/Scripts/python src/cutoff_test.py
.venv/Scripts/python src/check_qqp_overlap.py
.venv/Scripts/python src/human_check.py score
.venv/Scripts/python src/human_check.py review
```

Or open `DA6024_Execution.ipynb` and run all cells.

## Full pipeline, in order

Run each from this folder with `.venv/Scripts/python`. Steps marked **locked** produced files we froze; rerunning them would change the sentence IDs, so don't.

| # | Step | Command | Output | Needs |
|---|---|---|---|---|
| 1 | Download data | `src/download_data.py` | `data/paws_*`, `qqp_*`, `mrpc_*` | internet |
| 2 | Base sentences (**locked**) | `src/make_base_sentences.py` | `data/base_sentences.csv` | |
| 3 | Level 1, WordNet (**locked**) | `src/level1_synonyms.py` | `data/level1.csv` | |
| 4 | Levels 2 to 4, Qwen | `src/levels_llm.py 150` | `data/level2.csv` to `level4.csv` | Groq key |
| 5 | Level 5, MarianMT | `src/level5_marian.py 150 zh` | `data/level5_marian_zh.csv` | |
| 6 | Word overlap per level | `src/check_levels.py` | printed | |
| 7 | NLI meaning score | `src/meaning_check.py` | `data/meaning_check.csv` | |
| 8 | Apply meaning review, build test pairs | `src/make_pairs.py` | `data/ladder.csv`, `data/meaning_review.csv` | |
| 9 | Train models, tune cutoffs | `src/tune_thresholds.py` | `results/models.joblib`, `results/thresholds.json` | |
| 10 | Score all local methods | `src/run_tests.py` | `results/predictions_local.csv` | |
| 11 | Timing | `src/time_methods.py` | `results/timing_local.json` | |
| 12 | LLM judge | `src/judge.py gemini31`, then `src/judge.py gemini` (cross-check) | `results/judge_*.jsonl` | Gemini key |
| 13 | Tables and figures | `src/analyze.py` | `results/` | |
| 14 | Cutoff test | `src/cutoff_test.py` | `results/cutoff_test.csv` | |
| 15 | QQP overlap check | `src/check_qqp_overlap.py` | `results/qqp_overlap.json` | |
| 16 | Human check | `src/human_check.py score`, then `review` | `results/human_check*.csv` | |

Notes:
- Everything uses the fixed seed 42, so reruns give the same samples.
- `src/judge.py` saves each reply as it goes and resumes where it stopped. The free Gemini tier allows about 500 calls a day, so a full run takes two days.
- `src/judge.py qwen` is the self-preference test, which was not completed (see the report's Limitations).
- `src/level5_backtranslate.py` and `data/level5.csv`, `level5_da_zh.csv`, `level5_marian_da.csv` are earlier Level 5 attempts that we didn't use.
- `src/make_pairs.py` holds the meaning-review decisions as lists of rejected sentence IDs per level.
