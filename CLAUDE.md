# Paraphrase robustness — final year project

Plan: `docs/PLAN.md`. Status, decisions and log: `docs/PROGRESS.md` — read both at the start of every session.

## Keep docs/PROGRESS.md up to date (always)
After every step, add a note to `docs/PROGRESS.md` before ending the turn:
- **Log** (under today's date): what was done — scripts written/changed, files created, results/numbers.
- **Decisions**: every discussion that ended in a choice — what the options were, what was chosen, and why. Include choices to skip or postpone something.
- **To do later**: anything postponed or noticed but not fixed.
- **Current status / Open questions**: update when they change.

Write short, plain notes. Don't wait to be asked.

## Rules
- Use the venv: `.venv\Scripts\python <script>`. Set `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`.
- `data/base_sentences.csv`, `data/level1.csv`, `data/level1_review.csv` are locked — don't regenerate them.
- Discuss design changes with the user before building them.
- Keep code simple, few comments, written in a human tone.
