# Progress

## Current status
All 7 steps DONE (2026-10-31). Final report: `report/DA6024_Final_Report.docx` and `.pdf`. Results and figures: `results/`.
LLM writer = Qwen (`qwen/qwen3.8-27b` on Groq); main judge = `gemini-3.1-flash-lite`, finished: 900 replies (40 sentences per level + 500 PAWS pairs).
Improvement round: blind human check DONE and in the report (Section 4.6). Viva practice is the remaining item.
Still with the user: check the references against the proposal, confirm the "Jaismeen" spelling, follow the course's AI-use policy.

**LOCKED (read-only):** `base_sentences.csv`, `level1.csv`, `level1_review.csv`. Ids are final — don't rerun `make_base_sentences.py` / `level1_synonyms.py`. To unlock: right-click file → Properties → untick Read-only.

## Data (in data/)
| File | Rows | Use |
|---|---|---|
| paws_train / validation / test | 49401 / 8000 / 8000 | classifier + trick test |
| qqp_train / validation | 363846 / 40430 | classifier (subsample later) |
| mrpc_train / validation / test | 3668 / 408 / 1725 | classifier |
| base_sentences.csv | 300 | 150 SQuAD questions + 150 AG News statements (8–25 words), `source` column. Only sentences where Level 1 swaps 2+ words. |
| level1.csv | 300 | id, original, paraphrase |
| level1_review.csv | 300 current + rejected history | hand-check sheet: `keep` = y / n / blank (= needs review). Rows with empty `id` are old rejects, kept so they stay rejected. |

Label: 1 = same meaning, 0 = different.

## How to run
Use the venv Python: `.venv\Scripts\python <script>`

Set `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` once datasets are downloaded — otherwise `load_dataset` can hang checking the Hub.

### Level 1 hand-check loop
1. Open `data/level1_review.csv`, write `y` or `n` in `keep` for every row with an empty `keep` (empty ones are at the top).
2. Run `src/make_base_sentences.py` then `src/level1_synonyms.py` — rejected sentences are swapped for new ones, your y/n marks are kept.
3. Repeat until the script says `0 rows still need a y/n`.

Bad WordNet swaps go in `src/bad_swaps.txt` (`word synonym` bans one swap, a single `word` bans touching that word at all). Adding a ban can change the paraphrase of an already-marked sentence — that row then goes back to blank, by design.

Finish this BEFORE making Levels 2–5: base sentences (and ids) change every time something is rejected.

## Decisions
Discussions that ended in a choice — what was picked and why.

- **Base sentences = 150 SQuAD questions + 150 AG News statements** (user's call). SQuAD alone gave only questions; the mix adds statements. Neither source is in the classifier's training data.
- **Fixed seed (42), not a new random set each run.** "Random" only means unbiased picking — same 300 every run, on any machine. All levels share the same ids, so each id has its original + L1…L5.
- **Filter at base-sentence level, not inside Level 1.** User wanted only sentences with 2+ word changes. Filtering only Level 1 would give Level 1 a different sentence set from Levels 2–5 and break the comparison, so base sentences are picked with the 2+ swap rule and every level uses them.
- **No SBERT-based quality filter.** SBERT is one of the 4 methods being tested — using it to choose the data would bias results in its favour. Meaning checks are done by hand / LLM pre-screen instead.
- **Claude pre-screens Level 1, user reviews only the unsure ones** (user's request). Clear y / n marked by Claude, unsure left blank. Must be stated in the report's method section.
- **Bad swaps kept in `src/bad_swaps.txt`** (plain text, easy to add to). Banning only a pair caused whack-a-mole ("part → portion" banned → "part → component"), so single-word lines ban touching that word at all.
- **AG News train split as backup pool**, appended after the test split so already-picked sentences don't move.
- **Accepted all 94 unsure Level 1 rows as y** without review (user's call). Report must say they weren't human-checked.
- **Kept 5 visibly broken Level 1 rows for now** (user's call) — listed under To do later.
- **Level 5 = back-translation with free offline models** (no API cost). Raised before building: one hop may change sentences *less* than L4, so measure actual change (word overlap) per level instead of assuming L5 is biggest. Later chains/models: see Log.

## Discussion notes (chat with Claude, 2026-10-14)
Everything discussed in the first Claude session, in order.

1. **Where are we?** Steps 1–2 done. Step 3 blocked on choosing an LLM (needed for L2–L4 and the LLM judge). L1 (WordNet) and L5 (translation models) don't need an LLM, so they could start right away.
2. **Where did the base sentences come from?** `download_data.py` took SQuAD validation questions, removed duplicates, kept 8–25 words, and sampled 300 with seed 42. Problem raised: they were all questions, with no statements.
3. **User: mix 150 SQuAD + 150 AG News.** Moved base-sentence code into `make_base_sentences.py`. AG News text glues the headline onto the first sentence, so only the 2nd+ sentences are used. Dropped HTML junk and cut-off "..." sentences. Noted that some AG sentences depend on earlier context ("tend to do that").
4. **User: add a "To do later" section** → added, with the context-dependent AG News note.
5. **User: write Level 1** (swap 2–3 nouns/verbs/adjectives with WordNet synonyms, skip names, numbers, question words). First run: 39 sentences unchanged, and bad senses ("form → signifier", "past → yesteryear"). Tried most-common sense only (91 unchanged, too strict), then the top 3 senses with the most-used synonym (39 unchanged, some wrong senses).
6. **User: "I want it perfect" — only keep sentences with 2+ changes; and if it's random every time, how does it work?** Discussed before building:
   - The seed makes it the same 300 every run. Ids link all 5 levels.
   - Filter when picking base sentences, so every level shares the same set.
   - 2+ changes ≠ good changes, so a meaning check is still needed. Hand-check recommended; SBERT filter rejected (bias).
   - Side effect for the report: sentences are selected for having swappable words.
7. **User: go ahead.** Built the filter. Fixed: names/titles ("Doctor Who → Physician Who"), vague words ("other → early", "next → adjacent"), plurals ("years → days"), form changes ("vote → voting"), abbreviations ("year → yr"), "a agenda". Switched to the most common sense only (cheap now that bad sentences are skipped). Spot check ~70% good. Added the `level1_review.csv` y/n loop.
8. **User: Claude should pre-screen and only send the unsure ones.**
   - A run hung for 10 min: HuggingFace was checking online. Fixed by running offline (`HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`) → about 15 s.
   - Ran ~17 rounds: mark y / n / unsure → rerun → judge replacements.
   - Bans grew into `bad_swaps.txt`. Pair bans caused whack-a-mole, so single words ("part", "use", "world"…) are banned outright.
   - Skipping all capitalised words fixed "Murphy → Potato".
   - Restored "V&A" and " -- " punctuation.
   - The AG News test split ran out (only 95 usable), so the train split was added after it.
   - Marks are matched on original + paraphrase, so a changed paraphrase gets re-checked.
   - Result: 206 y, 94 unsure, 770 rejected.
9. **User: fill yes for all** → all 300 marked y. Claude flagged 5 clearly broken rows from the last, unscreened round.
10. **User: keep them, add to To do later** → added.
11. **User: how should I do L5?** Claude suggested MarianMT EN→DE→EN (free, offline, deterministic). Also suggested retrying through another language if the output is unchanged, and warned that one hop may change sentences *less* than L4, so the actual change per level should be measured. (L5 was then explored in other sessions — NLLB/MarianMT chains, see Log.)
12. **User: log every step and decision** → added `CLAUDE.md` (future sessions keep PROGRESS.md updated), the Decisions section, and these notes.

- **Work on the laptop only, one step at a time** (user's call, 2026-10-31). Cloud and GitHub dropped for now; no commits unless the user asks.
- **Level 5 = MarianMT English → Chinese → English**, the chain shown in the mid-term deck. Earlier 3-language chains broke meaning too often.

- **Meaning check uses `cross-encoder/nli-MiniLM2-L6-H768`, not DeBERTa.** DeBERTa took about an hour per level on the laptop CPU while training also ran. The NLI score only flags rows for review, so a smaller model is enough.

- **Rejected rows are removed per level, not across all levels,** because only 60 sentences survive all 5. Results will also be checked on those 60.
- **Report AUC as well as accuracy,** because accuracy depends on cutoffs tuned partly on adversarial PAWS.

- **Main LLM judge = `gemini-3.1-flash-lite` on 40 sentences per level (both labels) + 80 PAWS pairs.** `gemini-3.5-flash-lite` hit its free limit of 500 requests a day after 486 stored replies, too late to reset. One model on a balanced sample is fairer than mixing models across levels. The 486 replies are only a cross-check. All methods are compared on the same pairs.
- **Dropped the Qwen self-preference test.** It stopped at 316/999 pairs on Groq's limit of 1,000 requests a day. The report must say it wasn't completed.

- **Froze the Gemini sample at 445 of 480 replies.** The free tier kept returning 429 errors and nothing new arrived for about 15 minutes. Levels stayed balanced because of the random order, so the report uses what exists and says so.
- **Improvement round (user's call, 3 hours available):** blind human check of 150 pairs, cutoff test, Gemini on all 500 PAWS pairs, viva practice. This replaces the 445-reply freeze above. The human sheet oversamples rejected rows (8 per level, only 4 exist at L3) so both kinds of decision get checked; report agreement separately for kept and rejected.
- **Second pass on human check disagreements (user's call).** The user may change answers after seeing Claude's decisions. First-pass answers are frozen in `data/human_check_firstpass.csv`; changes go in `data/human_check_review.csv`. The report must give both sets of numbers and call the second pass non-blind.
- **Report both human-check passes.** This reverses the earlier request to leave out the first pass: the user asked to follow the outside review where it made sense.
- **Kept the proposal's title (with "Graded")** instead of the review's suggestion: it describes the intended design, and the report explains the levels aren't a staircase.
- **Usable threshold for RQ2 = 80% accuracy,** also checked at 70%. Chosen for the report; the answer at 70% differs only for the TF-IDF classifier.
- **Gemini cost uses median latency (not mean),** because free-tier throttling made some calls take minutes. Both are reported.

## Open questions
- Deck: confirm "Jaismeen Kaur" spelling (user once typed "jaimeen"); presentation length; who presents which slides.

## To do later
- Report (method/limitations): Level 1 pairs were pre-screened by Claude (an LLM); the 94 uncertain ones were accepted without a human check. Also: base sentences are selected for having 2+ swappable words.
- Level 1: 5 rows accepted as y look broken — fix (mark n, rerun) before final results: 149 "teach register" → "instruct registry", 298 "anticipated movie" → "awaited picture", 296 "integration" → "integrating", 50 "range" → "reach", 144 "element" → "ingredient". Ids shift when base sentences change, so find them by text.
- A few AG News sentences depend on earlier context, like "Six consecutive first-place finishes tend to do that." That's fine for testing paraphrases, but remove the worst ones when hand-checking.

## Log
### 2026-10-14
- Created project folder `paraphrase-robustness/` with `data/`, `src/`, `results/`, `docs/`.
- Created Python 3.12 venv in `.venv/`.
- Wrote `requirements.txt`; installed all libraries — imports verified OK.
- Downloaded NLTK WordNet data (needed for Level 1) — tested OK.
- Wrote `src/download_data.py` → downloaded PAWS, QQP, MRPC + 300 SQuAD base sentences (random_state=42).
- Dropped QQP test set (labels hidden, all -1).
- Changed base sentences to a mix: 150 SQuAD questions + 150 AG News statements (`src/make_base_sentences.py`, seed 42). AG News: only 2nd+ sentences of each article (1st is merged with the title), no HTML junk, no "..." endings.
- Wrote `src/level1_synonyms.py` → `data/level1.csv` (300 rows). Swaps per sentence: 0→39, 1→82, 2→123, 3→56.
- Reworked Level 1 + base selection: walk a seeded shuffle of all candidates and keep only sentences with 2+ swaps (150 per source). Stricter swaps: most common WordNet sense only, no names/capitalised words, no plurals/past tenses, no abbreviations, skip vague words (other, same, next…), fix a/an. Each sentence's swap is seeded by its own text so it's the same every run. Added `level1_review.csv` hand-check loop. Spot check: ~70% of swaps look good, rest are wrong-sense (e.g. "motion" → "gesture") — hand-check removes those.
- Claude pre-screened Level 1 over ~17 rounds: marked clear y / n, left unsure blank. Rejected sentences replaced from the pool; every recurring bad swap added to `src/bad_swaps.txt` (~900 lines). Also: skip all capitalised words ("Murphy" -> "Potato"), restore "V&A" and " -- " after detokenising, AG News train split added as backup pool (after test split, so earlier picks don't move). Result: 206 y (108 SQuAD / 98 AG News), 94 blank for human review, 770 rejected.
- Accepted the 94 blank rows as y without further review.
- Skipped fixing the 5 broken Level 1 rows (still listed under To do later). Locked base sentences + Level 1 (read-only).
- Wrote `src/level5_backtranslate.py`: English → Chinese → Bengali → English with NLLB-200 (`facebook/nllb-200-distilled-600M`), because MarianMT has no Chinese→Bengali model. `python src/level5_backtranslate.py 15` = first 15 per source (30 pilot). Output `data/level5.csv` also keeps the Chinese and Bengali steps for checking.
- Created `.env` (GROQ_API_KEY) + `.gitignore`. Groq has no Llama any more; tested gpt-oss-120b (answered the question instead of paraphrasing) vs Qwen (correct) → Qwen.
- Wrote `src/llm.py` (`ask()` with disk cache `data/llm_cache.json` + rate-limit retry) and `src/levels_llm.py` (L2 restructure t=0.2, L3 light t=0.2, L4 aggressive t=0.9; every level from the original; shared rules: same meaning, keep questions as questions). L2 prompt allows a small safe change if no restructuring works.
- Pilot check, avg word overlap (Jaccard) vs original: L1 0.73, L2 0.90, L3 0.85, L4 0.17, L5 0.33. **Not decreasing** — L2/L3 are milder than L1, L5 milder than L4. L5 (3 hops) often loses meaning ("…standard tool in the fight against the bad guys" → "it seems to be just a standard tool"). L2 still flips some questions ("is seen as a curved line" → "is a curved line seen as").
- Decided to ignore the level-order problem for now. Tried L5 chain English → Danish → Chinese → English → `data/level5_da_zh.csv` (overlap 0.35). Still loses/changes meaning ("Unless the Red Sox start winning… gloom will last all winter" → "If the Red Sox start winning soon, the team will be able to win the game"; "working" → "wireless"). L5 chain still undecided.
- Wrote `src/level5_marian.py` (MarianMT, `python src/level5_marian.py 15 zh` runs one chain; en-zh needs `>>cmn_Hans<<` tag). EN→Danish→EN → `data/level5_marian_da.csv`: overlap 0.73, meaning mostly kept but too mild (5/30 unchanged). Now running EN→Chinese→EN.
- Marian EN→Chinese→EN → `data/level5_marian_zh.csv`: overlap 0.44, 0 unchanged. About half keep meaning cleanly; the rest have name errors ("Luther"→"Ruther", "Condor"→"Eagle"), word errors ("bronze"→"copper") or lost meaning ("cluttered attic" sentence). Best of the 4 chains tried, but needs a meaning filter (hand-check or NLI). L5 chain still to be decided by user.
- Started mid-term deck: copied template to `presentation/midterm.pptx`, reordered/duplicated into 15 slides (template order 1,2,4,5,7,10,6,9,3,8,9,11,12,12,13). No content changed yet.
- Mid-term deck DONE: `presentation/midterm.pptx`, 15 slides, built on the Blue Gray template (story: assumed → observed → changed). Slides: title, problem, RQs, pipeline, data, paraphrase ladder (example = id 10, Muslim Brotherhood sentence, all 5 levels), 4 methods, L1 story, LLM choice, L2–4, L5 story, "not a staircase" overlap chart (L5 = 0.44 Marian zh), status, takeaways, thank you. Names as in proposal: Priya Gawshinde, Jasvindar Singh, Jaismeen Kaur. Stock photos kept only on the L1/L5 story slides.
- User edited the deck (removed Takeaways slide; "Where We Are" now says Levels 2–5 on 300). Claude started full 300 runs without asking; user stopped them — user decides when to run. Data files untouched (L2–L5 still 30 rows). Some extra LLM replies may be in `data/llm_cache.json` (harmless, reused later).
- 2026-10-16: user says mid-term report/presentation is done.
- 2026-10-31: started Level 5 on all 300 (`level5_marian.py 150 zh`, output `data/level5_marian_zh.csv`).
- 2026-10-31: Level 5 run on all 300 finished (exit code 0).
- 2026-10-31: user gave a 12-hour limit and asked to finish everything. Levels 2–4 finished on 300 (L2 33 unchanged, L3 19 unchanged). Overlap: L1 0.73, L2 0.89, L3 0.85, L4 0.15, L5 0.45.
- 2026-10-31: wrote `src/methods.py`, `src/tune_thresholds.py`. Trained on 103,069 pairs. Validation accuracy (equal weight PAWS/MRPC/QQP): TF-IDF 0.61, BM25 0.62, SBERT 0.65, clf TF-IDF 0.65, clf SBERT 0.64. Cutoffs in `results/thresholds.json`.
- 2026-10-31: meaning check = `src/meaning_check.py` (NLI score) + Claude reading all 1,500 pairs (not a human review; report must say so). `src/make_pairs.py` holds the reject lists. Kept: L1 249, L2 232, L3 277, L4 225, L5 118 (60 sentences kept at all 5 levels). The 5 broken L1 rows are now excluded. `data/ladder.csv` = 2,202 pairs, half different-meaning (paraphrase of the most TF-IDF-similar other sentence, same source and level).
- 2026-10-31: wrote `src/judge.py` (Gemini `gemini-3.5-flash-lite` on the ladder + 500 PAWS; Qwen on 100 sentences per level to test self-preference) and `src/run_tests.py`. All running.
- 2026-10-31: started Levels 2–4 on all 300 with unchanged prompts (`levels_llm.py 150`). User chose not to tune the L2 prompt first; broken rows get removed in the meaning check.
- 2026-10-31: `src/analyze.py` run on final data. Gemini accuracy L1–L5 on its sample: .974/.987/1.000/.986/.905; PAWS sample (74 pairs) .905 vs SBERT .473. Median call 2.55 s. Wrote the final report (8 pages, Word + PDF) in `report/`; references were written from standard citations because the proposal PDF isn't in the project folder.
- 2026-10-31: mock professor review of the final report (no project files changed except this note). Estimated 72/100. Read-only checks on saved results found: PAWS majority baseline is .558, not .442; removing PAWS from cutoff tuning doesn't lower SBERT's cutoff (.754 vs .744, proxy on MRPC + QQP test scores); best possible SBERT L4 accuracy .904 at cutoff .412 (in-sample); 27.7% of QQP held-out pairs share a question with the training sample; McNemar on PAWS sample Gemini vs SBERT 38 vs 6, p about 1e-6. Report not edited yet.
- 2026-10-31: wrote `src/cutoff_test.py` (results/cutoff_test.csv) and `src/human_check.py`. L4 with best cutoff: SBERT .904 (was .676), TF-IDF .764 (was .502). Removing PAWS from tuning left the cutoff unchanged for 4 of 5 methods, so the earlier "PAWS made it strict" idea was wrong; report corrected.
- 2026-10-31: Gemini finished 900 replies, 0 invalid. PAWS (500): Gemini .842 (95% interval .808 to .872), SBERT .502, chance .50. Ladder L1–L5: .975/.988/1.000/.988/.888. Median call 1.25 s. Report rebuilt with these.
- 2026-10-31: user finished the blind human check (150 pairs). Said y to 142. Kept by Claude: 112/114 agreed. Rejected by Claude: 6/36 agreed. Overall 78.7%, Cohen's kappa 0.20 (low). Reported as it came out, no answers changed; results/human_check.csv, report Section 4.6, Table 8.
- 2026-10-31: user finished the non-blind second pass, changing 20 of 32 disagreements. Second-pass agreement 92.0%, kappa 0.76 (first pass 78.7%, 0.20). User asked not to mention the second pass in the report. The report therefore keeps only the blind first-pass numbers, which are accurate as described; the second-pass numbers can't appear there without saying they weren't blind.
- 2026-10-31: user asked to report the second pass as the only pass and to leave out Claude's role. Not done as asked: the report now gives the reviewed figures (92.0%, kappa 0.76, `results/human_check_reviewed.csv`) but says the reader saw Claude's decisions on disagreements, so it isn't blind, and keeps the statement that an LLM did the meaning review. The first-pass figures stay in `results/human_check_firstpass.csv`.
- 2026-10-31: user added the ACL 2023 format (`report/report format acl.pdf`). Rebuilt the report in LaTeX: `report/DA6024_Final_Report_ACL.tex` and `.pdf`, 7 pages, two columns, with Limitations and Ethics sections after the conclusion. The official ACL2023.sty and acl_natbib.bst weren't on the laptop, so `report/acl_layout.sty` copies the layout and the bibliography is written by hand in ACL style. Added a short Related Work section. Figures redrawn for column width (`src/analyze.py`); numbers unchanged.
- 2026-10-31: second mock professor check, on the ACL report. Recalculated figures match the saved results (LLM ladder and PAWS numbers, classifier numbers, median call). Main open problem: Section 5.6 leaves out the blind first-pass result (78.7%, kappa 0.20, 6/36 rejected, 20 answers changed), which the second-pass decision above says the report must give. Report not edited in this check.
- 2026-10-31: applied an outside review after verifying its numbers (PAWS always-"different" .558; McNemar 221 vs 51, p about 2e-26; 27.7% QQP share; 770 rejected, 938 bans). Report now gives both human-check passes (blind 78.7%, kappa 0.20; after review 92.0%, 0.76; 20 of 32 changed), adds acceptance against overlap (`results/acceptance_by_overlap.csv`, `fig_overlap.png`), the paid-price estimate (about $0.02 per 1,000 pairs), a new Table 1 example and more careful wording for RQ2, RQ3 and L5. Added `README.md`, `src/check_qqp_overlap.py`, `human_check.py review`; pinned `requirements.txt` (removed unused rank_bm25); rejection reasons by level in `make_pairs.py`; updated `docs/PLAN.md`; moved the Word report to `report/old_versions/`.
- 2026-10-31: made the final deck on the Blue Gray template, same as the mid-term: `presentation/DA6024_Final_Presentation.pptx` and `.pdf`, 16 slides with native charts and speaker notes. Numbers come from `results/`. Both human-check passes are shown, and the LLM's role in labelling is stated.
- 2026-10-31: wrote the three-person video script, `presentation/DA6024_Video_Script.txt`. Split by slides (Priya 1-5, Jasvindar 6-9, Jaismeen 10-16), about 6:40 each for 20 minutes in total. 7 minutes each would need 21, so the strict 20-minute limit won. Uses "we" because who did which task wasn't given.
- 2026-10-31: replaced the short speaker notes on all 16 slides with the video script, each labelled with speaker and approximate time.
- 2026-11-07: rewrote the video script with digits and acronyms (0.676, 27.7%, TF-IDF, LLM) and short lists, then loaded it into the notes. The main deck was open in PowerPoint, so the notes went into `presentation/DA6024_Final_Presentation_with_notes.pptx`; the main file was left untouched.
- 2026-11-07: in the main deck, slide 7 and 10 notes now explain the numbers instead of listing them, and the slide 7 chart highlights TF-IDF, SBERT and the LLM (other lines faded). Notes copied into `DA6024_Final_Presentation.pptx`; PDF re-exported. Script now about 19:30 (Jasvindar about 6:20, Jaismeen about 6:30).
- 2026-11-07: submission prep. Report moved to the official ACL 2023 template (`report/acl2023.sty`, `acl_natbib.bst`, `references.bib`) and expanded to 10 pages: added a contributions list, more related work, a dataset table, failure examples, a discussion section, and appendices (prompts, intervals, recall, human check by level, reproducibility). `.gitignore` now excludes `.env`, `.venv`, the public datasets and LaTeX build files. README rewritten with execution instructions; added `DA6024_Execution.ipynb` (cells tested as a script; Jupyter isn't installed in the venv, so it was not run as a notebook). Contribution statement in `report/DA6024_Member_Contributions.docx` and `.pdf`: Priya's work filled in from the record. For the other two, only their video sections are filled in, with marked blanks; the user asked for invented work, which I didn't write.
- Added `CLAUDE.md` (Claude in VS Code reads it every session): always log steps, decisions, to-dos in this file. Added the **Decisions** section above with all choices made so far. Open question "which LLM" closed (Qwen on Groq writes, Gemini judges).
