# Levels 2-4: LLM paraphrases (Qwen on Groq)
# Usage: python src/levels_llm.py [N]   (N = sentences per source, default all)

import os
import sys
import pandas as pd
from llm import ask

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL = "qwen/qwen3.8-27b"

RULES = (
    "Keep exactly the same meaning: do not add, remove, or answer anything. "
    "If it is a question, keep it a question. "
    "Reply with only the rewritten sentence."
)

LEVELS = {
    2: (0.2, "Rewrite this sentence by changing only its grammatical structure: "
             "reorder clauses or phrases, or switch between active and passive voice. "
             "Keep the same words wherever possible. "
             "If no restructuring keeps the exact meaning, make only a small safe change "
             "in word order instead. "),
    3: (0.2, "Paraphrase this sentence with minimal changes: "
             "reword only a few words or phrases and keep most of the original wording. "),
    4: (0.9, "Paraphrase this sentence aggressively: use different vocabulary and a different "
             "sentence structure, reusing as few of the original words as possible "
             "(names, numbers and technical terms may stay). "),
}


def clean(text):
    text = text.split("</think>")[-1].strip()      # drop thinking, if the model adds any
    text = text.splitlines()[0].strip()             # first line only
    return text.strip('"').strip("“”").strip()


base = pd.read_csv(os.path.join(DATA, "base_sentences.csv"))
if len(sys.argv) > 1:
    base = base.groupby("source").head(int(sys.argv[1]))  # first N of each source

for level, (temp, instruction) in LEVELS.items():
    rows = []
    for i, (sid, sent) in enumerate(zip(base["id"], base["sentence"])):
        prompt = f"{instruction}{RULES}\n\nSentence: {sent}"
        rows.append({"id": sid, "original": sent, "paraphrase": clean(ask(prompt, MODEL, temp))})
        print(f"  level {level}: {i + 1}/{len(base)}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, f"level{level}.csv"), index=False)
    print(f"saved data/level{level}.csv")
