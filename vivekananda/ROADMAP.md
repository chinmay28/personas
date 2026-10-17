# Roadmap: future improvements

State as of 2026-10-17: two QLoRA runs trained and evaluated. Run 2 (`vivekananda:8b-run2`) fixes the run 1
regressions and matches or beats the prompted `qwen3:14b` cascade on routing, quote discipline, persona marking
and speed, but trails it on oracle relevance (0.919 vs 0.989) and stance-probe agreement (0.74 vs 0.86). The
server default is still the prompted cascade. This file lists what could be done next, grouped by the problem
it addresses, with the evidence from `eval/results/` that motivates each item. Items marked **owner** need a
decision before work starts.

## 1. Decisions pending (owner)

- **Shipping model.** Ship run 2 as the single 8B, keep the prompted 14B cascade, or run a third training
  pass first (section 2). Run 2 is faster and safer; the 14B picks better passages.
- **Publish weights and dataset.** Hugging Face and Ollama accounts for `vivekananda:8b-run2` (Apache-2.0)
  and the CC0 training set were held on 2026-10-15. The GGUFs are 8.7 GB and 5.0 GB.
- **Add the "Life of Swami Vivekananda by his Eastern and Western Disciples"** for biographical scholar
  coverage. Today the scholar can only recount what the Complete Works record; questions about his life
  outside his own words get "no record". It is a third-person source, so it would need a separate genre
  that scholar may cite but oracle may never quote as his words.
- **Citations mandatory or on request.** Default is mandatory in every mode.
- **Persona exclusions.** Any topics persona should not extrapolate on at all. Current policy: nothing is
  banned.
- **Phone acceptance walk-through** from the plan's phase 9: install the PWA over the LAN, ask ten questions
  across modes, open five citations. Not yet done by anyone.

## 2. Closing the gap to the prompted baseline (candidate run 3)

Run 2 lost 8 oracle answers to "true but tangential" passages and 9 probes to stance disagreement. Both are
selection problems, not voice problems. Data changes first, hyperparameters last, as before.

- **Better expansion queries.** The run 2 model writes its own retrieval queries (tier 9), but the queries
  it produced on the misses are generic restatements ("religion and religions distinction", "Chitta and
  Vrittis definition"). The 14B's expansions carry more concrete terms. Options: generate tier 9 targets with
  retrieval-in-the-loop (keep only query sets that actually rank the gold paragraph in the top 5), and
  double the tier to 2,000.
- **Oracle plan selection on stance questions.** Probes such as "What is the root of weakness and sin?" and
  "What is the Gita's central teaching?" retrieved the right section but the plan chose a neighbouring
  paragraph. A small tier of hard negatives: blocks that contain both the stance passage and a plausible
  distractor from the same work, with the plan pointing at the stance passage. The 50 probes themselves must
  stay out of it.
- **Persona-versus-oracle boundary.** Three persona-expected questions went to oracle ("Is it wrong to want
  to be rich?", "What is the purpose of exercise and fitness?", "How do I know if my teacher is genuine?").
  The cascade prefers oracle when a passage exists, so these may be acceptable; if not, tier 6 needs
  questions whose block contains a near-miss passage with a persona target that cites it.
- **Third-person self-references.** Three in run 2 (probes 06, 46, 50), one in the baseline. Audit the tier 4
  source paragraphs for narrated passages that the "Swamiji said" filter missed, and add the regex used by the
  eval to the tier 4 builder.
- **Reranker over the top 50.** Proposed in phase 2b and deferred. The PyTorch environment now exists; a
  `bge-reranker-v2-m3` pass on the block before it is shown to the model raises the ceiling for every mode at
  the cost of roughly a second per answer. Measure on the 2,000-question retrieval set first.
- **One epoch or a lower rank.** Run 2's train loss (0.24 to 0.30 at the end) is well under the holdout loss
  (0.36) and the eval loss plateaued after step 500. If a run 3 is data-identical, try a single epoch to save
  three GPU hours and compare.

## 3. Evaluation quality

- **Judge reliability.** The `qwen3:14b` judge calls "I told the audience at Chicago" third person and marks
  several correct stances as disagreement (probe 28 on education in run 1). Options: a stronger local judge
  for the first-person and stance verdicts only, a two-judge majority, or a hand-labelled 50-item calibration
  set so judge drift is measurable between runs.
- **Relabel the decline extras.** The 10 "decline" questions in `eval/questions.jsonl` still carry that label
  from before the no-refusals decision; `eval_router.py` counts them as misses. Relabel to persona.
- **LLM-judged retrieval metric.** The gold-paragraph recall (0.70 at 5) is a pessimistic floor because the
  corpus answers the same question in many places. The phase 2b proposal of a "top 5 contains an equivalent
  passage" judgment on 200 questions was never run.
- **Mixed mode.** Not trained and not evaluated beyond accepting scholar or persona for the 10 mixed
  questions. Either define a two-part target and train it, or drop the label.
- **Human spot-check protocol.** A fixed 30-answer sample per run, read by the owner, with a short rubric
  (voice, grounding, usefulness). The automatic numbers cannot see register and tone.

## 4. Retrieval and corpus

- **Letters are the weakest genre** (0.57 recall at 5). Index letters with their recipient and date as a
  header prefix, as was done for section titles, and test on the letter subset.
- **Cross-volume duplicates.** Some lectures appear in more than one volume; the plan called for
  deduplication of near-identical passages and only the training set dedups by question. A paragraph-level
  near-duplicate map would stop the oracle from citing two copies of the same text as "see also".
- **Sanskrit term normalisation.** The FTS tokenizer strips diacritics, but transliteration variants
  (Jnana, Gyana; Shakti, Sakti) are not linked. A small synonym table applied to queries is cheap.
- **Endnotes.** The 430 editorial endnotes are extracted but unused. They explain names and terms and could
  be shown in the citation drawer.

## 5. Serving and clients

- **Make the backend switch a per-request choice** in the PWA settings so run 2 and the prompted cascade can be
  compared side by side on the same question from the phone.
- **Streaming for oracle and scholar.** Only persona prose streams token by token; the JSON modes wait for
  the full plan. Stream the status and mode events earlier and render quotes as each plan entry resolves.
- **Section browser.** A read-only route to browse volumes, works and sections, so a citation can be read in
  context beyond the single section drawer.
- **Export and share an answer** as plain text with citations from the PWA.
- **Authentication and HTTPS** before the server is reachable from outside the home network. The plan
  explicitly scoped auth out for the LAN.
- **Concurrency.** The server is a single process; two phone users will queue. Fine for now, worth a note.

## 6. Training pipeline hygiene

- **GPU reservation.** Training stops Ollama models at launch but a LAN client can reload one mid-run and
  cause an out-of-memory failure. Options: an Ollama `OLLAMA_MAX_LOADED_MODELS=0` toggle for the duration,
  or run training with a watchdog that pauses on VRAM pressure and resumes from the last checkpoint.
- **Dataset stats gate.** `run_run2_train.sh` should refuse to start if any tier in `data/dataset_stats.md`
  is under an expected minimum; the tier 9 dedup bug would have been caught automatically.
- **Reproducibility manifest** per run: dataset hash, tier counts, git commit, base model revision, Unsloth
  and transformers versions, written next to `metrics.md`.
- **Resume test.** `--resume` exists but has never been exercised on a real interruption.

## 7. Generalising to the next persona

The repository is named for more than one figure. What is Vivekananda-specific today and would need
abstracting:

- `extract_corpus.py` genre rules and letter metadata parsing (the epub splitter itself is generic).
- `traindata.py` bridge templates and the scholar question templates ("What did you write to X in ...").
- The persona preamble, the system prompt, and the 124 tier 6 topics.
- The eval set and probes, which must be written per figure by someone who knows the sources.
- Citation format `[Vol N, Work, Title]`, which assumes a multi-volume collected works.

A second persona with a smaller, single-volume corpus would be the right test of the abstraction, and would
show whether the prompted cascade alone is enough when the corpus is small.
