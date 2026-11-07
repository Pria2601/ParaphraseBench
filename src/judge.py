# LLM judge: asks a model "same meaning?" for each pair. Every reply is appended to a file, so a stopped run resumes.
# Usage: python src/judge.py gemini31          (main judge: 40 sentences per level + 500 PAWS test pairs)
#        python src/judge.py gemini            (whole ladder + 500 PAWS pairs; stopped at the daily limit, used as a cross-check)
#        python src/judge.py qwen              (100 sentences per level, same-meaning and different-meaning)

import json
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
SEED = 42
PROMPT = "Do these two sentences have the same meaning? Answer only yes or no.\nSentence 1: {}\nSentence 2: {}"


def gemini_call(model="gemini-3.5-flash-lite"):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(temperature=0, max_output_tokens=20)

    def call(prompt):
        r = client.models.generate_content(model=model, contents=prompt, config=config)
        u = r.usage_metadata
        return r.text or "", u.prompt_token_count, u.total_token_count
    return call


def qwen_call():
    from groq import Groq
    client = Groq()

    def call(prompt):
        r = client.chat.completions.create(model="qwen/qwen3.8-27b", temperature=0,
                                           messages=[{"role": "user", "content": prompt}])
        text = r.choices[0].message.content.split("</think>")[-1].strip()
        return text, r.usage.prompt_tokens, r.usage.total_tokens
    return call


def paws_sample():
    # same 500 pairs for every method, half of each label
    paws = pd.read_csv(os.path.join(DATA, "paws_test.csv"))
    sample = paws.groupby("label").sample(250, random_state=SEED)
    return pd.DataFrame({"set": "paws", "level": 0, "id": sample["id"].astype(str), "s1": sample["sentence1"],
                         "s2": sample["sentence2"], "label": sample["label"]})


def pairs_for(name):
    ladder = pd.read_csv(os.path.join(DATA, "ladder.csv"))
    ladder["set"] = "ladder"
    ladder["id"] = ladder["id"].astype(str)
    columns = ["set", "level", "id", "s1", "s2", "label"]
    if name == "gemini":
        return pd.concat([ladder[columns], paws_sample()], ignore_index=True)
    if name == "gemini31":
        # the free tier allows about 500 calls a day, so this ran over two days:
        # 40 sentences per level (both labels) + 80 PAWS pairs, then the rest of the 500 PAWS pairs
        picks = []
        for _, group in ladder.groupby("level"):
            ids = group["id"].drop_duplicates().sample(40, random_state=SEED)
            picks.append(group[group["id"].isin(ids)][columns])
        return pd.concat(picks + [paws_sample()], ignore_index=True)
    picks = []
    for _, group in ladder.groupby("level"):
        ids = group["id"].drop_duplicates().sample(100, random_state=SEED)
        picks.append(group[group["id"].isin(ids)])
    return pd.concat(picks, ignore_index=True)[columns]


name = sys.argv[1]
call = {"gemini": gemini_call, "gemini31": lambda: gemini_call("gemini-3.1-flash-lite"), "qwen": qwen_call}[name]()
pairs = pairs_for(name)
os.makedirs(RESULTS, exist_ok=True)
path = os.path.join(RESULTS, f"judge_{name}.jsonl")

done = set()
if os.path.exists(path):
    with open(path, encoding="utf-8") as f:
        done = {json.loads(line)["key"] for line in f}

keys = pairs["s1"] + "\t" + pairs["s2"]
todo = pairs[~keys.isin(done)].drop_duplicates(["s1", "s2"])
# random order keeps the levels balanced if the run has to stop early (the two labels of a sentence stay together)
order = todo[["set", "level", "id"]].drop_duplicates().sample(frac=1, random_state=SEED)
todo = order.merge(todo, on=["set", "level", "id"], sort=False)
print(f"{name}: {len(pairs)} pairs, {len(todo)} to do", flush=True)

with open(path, "a", encoding="utf-8") as out:
    for count, r in enumerate(todo.itertuples(), 1):
        wait = 5
        while True:
            try:
                start = time.time()
                text, prompt_tokens, total_tokens = call(PROMPT.format(r.s1, r.s2))
                seconds = time.time() - start
                break
            except Exception as e:
                message = str(e)
                daily = "per day" in message.lower() or "PerDay" in message
                print(f"  error after {count - 1} calls: {message[:200]}", flush=True)
                if daily or wait > 600:
                    print("stopping; rerun later to resume", flush=True)
                    sys.exit(2)
                time.sleep(wait)
                wait *= 2
        out.write(json.dumps({"key": f"{r.s1}\t{r.s2}", "reply": text, "seconds": seconds,
                              "prompt_tokens": prompt_tokens, "total_tokens": total_tokens}, ensure_ascii=False) + "\n")
        out.flush()
        if count % 100 == 0:
            print(f"  {count}/{len(todo)}", flush=True)
print("done", flush=True)
