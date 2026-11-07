# Small wrapper around the Groq API with a disk cache (same prompt = no new API call)

import json
import os
import time
from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()
client = Groq()
CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "llm_cache.json")
cache = json.load(open(CACHE_FILE, encoding="utf-8")) if os.path.exists(CACHE_FILE) else {}


def ask(prompt, model, temperature):
    key = f"{model}|{temperature}|{prompt}"
    if key in cache:
        return cache[key]
    for wait in [5, 15, 30, 60, 120]:
        try:
            r = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except RateLimitError:
            print(f"  rate limited, waiting {wait}s", flush=True)
            time.sleep(wait)
    else:
        raise RuntimeError("still rate limited after 5 retries")
    answer = r.choices[0].message.content.strip()
    cache[key] = answer
    json.dump(cache, open(CACHE_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return answer
