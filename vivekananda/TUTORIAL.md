# Tutorial: building AI Vivekananda from the Complete Works, step by step

This is the narrative walkthrough of how the Swami Vivekananda persona in this repository was built, in the
order it was actually done, including the first fine-tune that failed and the second one that fixed it.
`README.md` is the terse runbook with the exact numbers per phase; `PLAN.md` is the design document.
This file explains *why* each step exists and what to watch for if you repeat it for another figure.

Everything below ran on one machine: Ubuntu 22.04, an RTX 4080 with 16 GB of VRAM, Python 3.12 via `uv`,
and Ollama serving the helper models. Nothing leaves the machine. Total wall time, excluding the two
six-hour training runs, was about four working days.

Contents

1. [The idea and the one hard rule](#1-the-idea-and-the-one-hard-rule)
2. [Environment](#2-environment)
3. [Extracting the corpus from the epub](#3-extracting-the-corpus-from-the-epub)
4. [Building the hybrid retrieval index](#4-building-the-hybrid-retrieval-index)
5. [Measuring retrieval with generated questions](#5-measuring-retrieval-with-generated-questions)
6. [The three modes, prompted on a stock model](#6-the-three-modes-prompted-on-a-stock-model)
7. [The evaluation set and the stance probes](#7-the-evaluation-set-and-the-stance-probes)
8. [The router](#8-the-router)
9. [Generating the training dataset](#9-generating-the-training-dataset)
10. [Fine-tuning with QLoRA (run 1)](#10-fine-tuning-with-qlora-run-1)
11. [Exporting to GGUF and Ollama](#11-exporting-to-gguf-and-ollama)
12. [Evaluating the fine-tuned model against the prompted baseline](#12-evaluating-the-fine-tuned-model-against-the-prompted-baseline)
13. [Diagnosing run 1 and redesigning the data (run 2)](#13-diagnosing-run-1-and-redesigning-the-data-run-2)
14. [Run 2 results](#14-run-2-results)
15. [Serving it: API, CLI and PWA](#15-serving-it-api-cli-and-pwa)
16. [Lessons that transfer to the next persona](#16-lessons-that-transfer-to-the-next-persona)

---

## 1. The idea and the one hard rule

The goal is a local system that answers as Swami Vivekananda, in the first person, from the nine-volume
Complete Works (public domain, about 8 million characters). A stock language model already "knows"
Vivekananda from pretraining, so the point of fine-tuning is not knowledge. It is voice, discipline about
what counts as a quotation, and a fixed output contract that the serving code can verify.

The hard rule that shaped every decision: **anything presented as a quotation must exist verbatim in the
corpus.** A fabricated quote attributed to a revered figure is the worst possible failure, so the design
makes it structurally impossible rather than merely unlikely.

That rule leads to three *grounding levels*, all spoken as him, picked per question by a router:

| mode | what it does | how grounding is enforced |
|---|---|---|
| **scholar** | he recounts what he said or wrote; every sentence carries a citation | output is a JSON list of sentences with paragraph ids; sentences whose ids do not resolve to a retrieved passage are dropped |
| **oracle** | the answer *is* his own passage, quoted | the model outputs a *selection plan* (paragraph id, sentence range); the renderer copies text from the corpus, so the quote is verbatim by construction |
| **persona** | he extrapolates from his principles when the corpus is silent | the answer must open with an explicit marker ("I did not speak to this directly, but...") and any quoted span is checked against the corpus |

The cascade prefers scholar over oracle over persona, because that is the order of decreasing
verifiability. A **quote guard** runs on every answer regardless of mode: any span in quotation marks must
fuzzy-match a corpus sentence at 0.95 similarity or the marks are removed.

Two owner decisions made early and never revisited: no refusal training and no "after 1902" filters
(questions about later events route to persona, which marks them as extrapolation), and the oracle quotes
only the key paragraph (about 250 words) and cites the rest.

## 2. Environment

```bash
git clone https://github.com/chinmay28/personas.git && cd personas
uv sync                      # retrieval, modes, server, CLI, eval
uv sync --extra train        # PyTorch (CUDA), Unsloth, TRL, PEFT, bitsandbytes, gguf: only for section 10 onward
ollama pull bge-m3           # embedding model for retrieval
ollama pull qwen3:14b        # helper model: question generation, data synthesis, judge, prompted baseline
cd vivekananda
```

All scripts are run as `.venv/bin/python scripts/<name>.py` from the `vivekananda/` directory. The training
extras are heavy (several GB of CUDA wheels); they were deliberately left out until the dataset existed.

A 27B model was tried for data synthesis and rejected: it ran at 13.5 tokens per second against 68 for the
14B on this GPU, and when loaded alongside anything else it spilled to CPU. The 14B is used for everything.

## 3. Extracting the corpus from the epub

```bash
VK_EPUB="/path/to/Complete Works of Swami Vivekananda.epub" .venv/bin/python scripts/extract_corpus.py
```

The epub is not in the repository; the extracted text in `corpus/` is. Two scripts do the work:

- `scripts/epub_split.py` is a generic epub-by-table-of-contents splitter. It walks the spine in document
  order and assigns every block to the latest `toc.ncx` anchor, so a section can span several HTML files.
  It treats a double `<br>` inside one `<p>` as a paragraph break (this epub uses both styles) and drops
  title lines that merely repeat the TOC label.
- `scripts/extract_corpus.py` adds the Complete Works specifics: genre from the work title (lecture,
  book_chapter, letter, dialogue, qa, interview, poem, notes, essay, sayings, ...), the volume from document
  position, and for letters the date as ISO, place, recipient and salutation. It also records sentence
  offsets for every paragraph, which the oracle needs later to quote sentence ranges.

Outputs: `corpus/sections.jsonl` (1,391 TOC entries), `corpus/paragraphs.jsonl` (20,694 paragraphs),
`corpus/endnotes.jsonl`, and an extraction report. Acceptance was checked by hand: every TOC entry
accounted for, total characters within 5% of the estimate, 50 random sections spot-checked for genre
(all correct), date present for 98.6% of letters and recipient for 100%.

Why paragraphs and not fixed-size chunks: the oracle quotes real paragraphs with real sentence boundaries,
and citations point at `[Vol N, Work, Title]`. Fixed chunks would cut sentences and lose the citation unit.

## 4. Building the hybrid retrieval index

```bash
.venv/bin/python scripts/build_index.py --model bge-m3
.venv/bin/python scripts/search.py "What is Karma-Yoga?" -k 8
```

`index/corpus.db` is a single SQLite file holding the metadata tables, FTS5 full-text tables for
paragraphs and sections, and one sqlite-vec table per embedding model. No server process.

Hybrid search is non-negotiable for this corpus: Sanskrit terms, names and places (Sânkhya, Alasinga,
Almora) defeat embeddings alone, and BM25 alone misses paraphrased modern questions. Dense and BM25
rankings are fused with reciprocal rank fusion. The FTS tokenizer strips diacritics so "Sankhya" matches
"Sânkhya".

`scripts/vklib.py` holds everything shared downstream: the DB connection, the citation formatter, the Ollama
embedding client, RRF, and the two search functions. Every later stage imports it, so retrieval behaves
identically in data generation, evaluation and serving.

## 5. Measuring retrieval with generated questions

You cannot tune retrieval without a labeled set, and nobody is going to hand-label thousands of
paragraphs. So the 14B wrote the labels: for each eligible paragraph (300 to 4,000 characters, in the
spoken and written genres) it generated two questions that the paragraph answers, one using the passage's
own terms and one in plain modern phrasing. Each question's gold answer is the paragraph it came from.

```bash
.venv/bin/python scripts/gen_questions.py --sample 1000      # seeded 1,000-paragraph sample, ~30 min
.venv/bin/python scripts/eval_retrieval.py                   # recall@k per model and method
.venv/bin/python scripts/tune_retrieval.py                   # fusion variants
```

The full run (12,926 questions over 6,463 paragraphs) was done with the datamill tool's
question-for-passage mode; `scripts/import_datamill_pairs.py` converts its export. One detail mattered:
telling the generator that the author is Vivekananda. Without that hint it wrote "What does the speaker
suggest..." questions.

Results on the 1,000-paragraph sample, paragraph-level recall of the gold paragraph:

| variant | recall@5 | recall@10 |
|---|---|---|
| nomic-embed-text + BM25 | 0.63 | 0.70 |
| bge-m3 + BM25 | 0.66 | 0.73 |
| bge-m3 on header-prefixed paragraphs + bge-m3 + BM25, RRF k=10 (chosen) | 0.70 | 0.77 |

The plan's target of 0.85 was not met, and that was accepted after reading the misses: the Complete Works
answer the same general question in many places, so an equally good passage often outranks the gold one.
The gold-paragraph metric is a pessimistic floor. Specific questions reach 0.76 and letters are the weakest
genre.

## 6. The three modes, prompted on a stock model

Before any training, the whole system was built as prompts on `qwen3:14b` (`scripts/modes.py`). This
prompted version is the development stand-in, the baseline the fine-tune must beat, and the fallback that
still runs today.

```bash
.venv/bin/python scripts/ask.py "What is Karma-Yoga?"                    # routed automatically
.venv/bin/python scripts/ask.py "..." --mode oracle                       # override the router
.venv/bin/python scripts/eval_modes.py --oracle 100 --scholar 50
```

The key pieces in `modes.py`:

- `retrieve`: hybrid search, optionally fused with two or three short keyword queries the model writes
  itself (query expansion). The block shown to the model is 8 paragraph hits plus up to 2 section openings,
  with every sentence numbered.
- `run_oracle`: the model returns `{"quotes": [{"para": "p-00412", "from": 2, "to": 5}], "bridges": [...],
  "cite_more": [...]}`. The renderer copies the sentences from the corpus, prints bridges as plain
  first-person sentences outside quotation marks, caps quoted words at 250, and lists the rest as
  "I say more on this in ...". An empty plan falls through to persona.
- `run_scholar`: the model returns `{"sentences": [{"text": ..., "ids": [...]}]}`. Sentences whose ids
  are not in the retrieved block are dropped. The first version used free prose and only 65% of sentences
  ended up cited; switching to the JSON list made every delivered sentence cited by construction.
- `run_persona`: first-person prose with `[p-xxxxxx]` tags rendered as numbered citations, tags outside the
  block removed, and a required extrapolation marker.
- `quote_guard`: the 0.95 fuzzy match on every quoted span.

Acceptance on the prompted system: oracle verbatim rate 100% over 164 quotes with zero over the cap.

Two infrastructure bugs found here that would have silently poisoned everything downstream:

- Ollama's OpenAI-compatible endpoint runs model slots with a 4,096-token context and silently truncates
  longer prompts. Retrieval blocks reach 6,200 tokens. All calls were moved to the native `/api/chat`
  endpoint with `num_ctx` set to 8,192.
- The quote regex paired the closing mark of one one-letter quote with the opening mark of the next (`the
  little "I" ... the great "I"`) and then stripped both. Spans must now start and end on a non-space
  character.

## 7. The evaluation set and the stance probes

`eval/questions.jsonl` has 150 questions, 50 per expected mode, plus 20 extras for the router.
`eval/position_probes.jsonl` has 50 documented stances (Advaita versus dualism, caste, women, Buddhism,
ritual, service, renunciation, science, education, Christ, Islam, the Vedas, politics, and more). Each probe
has the stance in one sentence, a probe question, and a support block with the paragraph id and excerpt that
documents it. Every support passage was retrieved from the index and read before being chosen.

This set is frozen before any training data is generated, and the dataset assembler drops any training
record whose question matches an eval question. Without that, the eval measures memorization.

## 8. The router

A prompted classifier returning `scholar | oracle | persona | mixed`, with the retrieved passages as
evidence. The first prompt scored 81%; the main confusion was oracle questions ("What is Maya?") sent to
scholar. Rewriting the prompt with explicit definitions brought it to 94%, above the 90% acceptance.

In production the router is not a separate call: the fine-tuned model emits `<mode>X</mode>` as its first
line, having seen the retrieval block. That is why the block format had to be fixed before the dataset
was built (section 9): the single model picks the mode after seeing the block, so the block cannot depend
on the mode.

## 9. Generating the training dataset

This is where most of the effort went, and where run 1 went wrong. Every record has the same shape:

```
system:    fixed prompt defining the three mode tags and their output formats (traindata.UNIFIED_SYSTEM)
user:      the retrieval block (8 hits + 2 section openings, sentences numbered) + "QUESTION: ..."
assistant: <mode>X</mode>
           the mode output (JSON plan for oracle, JSON sentence list for scholar, prose for persona)
```

The block is built with deterministic retrieval, no query expansion, so it is reproducible and the serving
code can build the exact same thing. Loss is computed on the assistant turn only.

Records come from tiers, highest fidelity first. `scripts/build_dataset.py tierN` builds one tier at a time
(resumable, logged); `scripts/assemble_dataset.py` verifies, caps, dedups and splits.

| tier | source | construction | mode |
|---|---|---|---|
| 1 | Questions-and-Answers sections, interviews | parsed on the `Q. —` / `A. —` markers; interview answers kept only if verbatim in one paragraph | oracle |
| 2 | Conversations and Dialogues | `Disciple:` / `Swamiji:` lines; the disciple's line is the question | persona in run 1, oracle in run 2 |
| 3 | Epistles (724 letters) | the 14B writes the *incoming* letter his reply answers; his letter is the target | persona in run 1, oracle in run 2 |
| 4 | Lectures, Yoga books | the question-for-passage set from section 5; the paragraph becomes an oracle selection plan | oracle |
| 5 | retrieved sections | templated questions ("What did you write to Mrs. Bull in June 1896?") answered by the prompted scholar; kept when citations verify | scholar |
| 6 | none | modern questions generated per topic, answered by the prompted persona, quote-guarded, marker required | persona |
| 8 | databricks-dolly-15k | general chat under a plain system prompt, 12% of the total, to keep instruction following | chat |
| 9 (run 2) | questions from the other tiers | question to `{"queries": [...]}`, so the model can expand its own retrieval | expand |

A refusal tier (7) was planned and removed at the owner's request: the model answers everything in his voice.

Details that took iteration:

- **Oracle targets are the plan, not the text.** For tier 1 and 4 the plan is built deterministically: the
  whole paragraph if it is under 200 words, otherwise the sentence window with the most overlap with the
  question. If retrieval missed the gold paragraph it is injected in place of the lowest hit (about 8% of
  tier 4), and the record is flagged `gold_injected`.
- **Synthesized incoming letters leaked the reply.** The first prompt produced letters that replied *to* his
  letter. The prompt now states the correspondent has not yet heard from him, and a regex rejects leaks.
- **Scholar prose was monotonous.** The 14B wrote "I told X that ..." in every sentence until the prompt
  asked it to name the recipient once and vary openings.
- **Every record is verified at assembly**: oracle plans reference only block paragraphs with valid
  sentence ranges and no more than 250 quoted words; scholar sentences cite only block paragraphs; persona
  tags point into the block; every quoted span passes the quote guard. Failures are dropped and listed in
  `data/tiers/assemble_dropped.jsonl`.
- **Holdout**: 5% of each tier, stratified, seeded, split before training.

```bash
.venv/bin/python scripts/build_dataset.py tier1     # and tier2 ... tier8, each resumable
.venv/bin/python scripts/assemble_dataset.py        # -> data/train.jsonl, data/holdout.jsonl, data/dataset_stats.md
.venv/bin/python train/tokencheck.py                # real tokenizer lengths; nothing may exceed 8,192
```

Run 1 ended with 6,195 training records: oracle 3,080, scholar 1,382, persona 1,278, chat 782. Only 6% of
the tokens are response tokens; the retrieval block dominates every record.

## 10. Fine-tuning with QLoRA (run 1)

Base model: `Qwen/Qwen3-8B`, the post-trained hybrid-thinking model, used in non-thinking mode throughout.
The instruct model rather than the base so JSON discipline and general chat start from something that
already follows instructions. Unsloth QLoRA: 4-bit base, LoRA rank 32, alpha 32, dropout 0, learning rate
2e-4 cosine with 3% warmup, 2 epochs, batch 1 with gradient accumulation 16, max sequence 8,192, Unsloth
gradient checkpointing, 8-bit AdamW.

```bash
.venv/bin/python train/train_qlora.py --smoke                      # 4 records, 2 steps: VRAM and label mask check
.venv/bin/python train/train_qlora.py --out train/out/run1         # the real run, ~6 h
```

The one thing to get exactly right is the **label mask**. Loss must fall only on the assistant turn, and the
assistant turn must be byte-for-byte what Ollama will put after `<|im_start|>assistant` at serving time.
For Qwen3 with thinking off that is `<think>\n\n</think>\n\n<mode>X</mode>\n...`. The smoke test asserts
the mask starts exactly at `<think>`; a 3,256-token oracle record has 52 loss tokens.

The smoke test also caught a scheduling hazard: another client on the LAN loaded a 27B model into Ollama
mid-test and took 13 GB of the GPU, which made training OOM. The training scripts now stop all Ollama models
first, and checkpoints every 100 steps with `--resume` limit the damage.

Run 1: 776 steps, 6.0 hours, 20.6 seconds per step, peak 9.4 GiB VRAM, masked holdout loss 0.588.

A measurement trap worth recording: the first metrics file also printed a "final holdout loss" of 2.35.
That was the raw holdout handed to the trainer's evaluate without the response mask, so it scored every
prompt token too. The script now masks the full holdout before the final evaluation.

## 11. Exporting to GGUF and Ollama

```bash
.venv/bin/python train/export.py --run train/out/run1 --name vivekananda:8b
```

The export merges the adapter into bf16 safetensors, converts with llama.cpp's `convert_hf_to_gguf.py`,
quantizes to Q8_0 and Q4_K_M, writes Modelfiles using the Qwen3 chat template copied from Ollama's own
`qwen3:14b` (`train/qwen3.template`) with `num_ctx 8192` and temperature 0.3, and runs `ollama create`.
Only the CPU build of `llama-quantize` is needed. Set `VK_LLAMA_CPP` if llama.cpp is not at `~/src/llama.cpp`.

One surprise: the merge downloads `unsloth/Qwen3-8B` (15 GB) even though `Qwen/Qwen3-8B` is cached, because
the adapter's recorded base is Unsloth's 4-bit variant. Budget the disk and the time.

Serving the fine-tuned model (`scripts/unified.py`) uses Ollama's `/api/generate` in raw mode with the prompt
rendered through the same template, verified equal to `tokenizer.apply_chat_template(..., enable_thinking=False)`.
Raw mode is needed because the chat endpoint cannot prefill an assistant turn, and a `--mode` override is
implemented as a prefilled `<mode>X</mode>` line. The model's output then goes through the *same renderers
and quote guard* as the prompted system.

## 12. Evaluating the fine-tuned model against the prompted baseline

```bash
.venv/bin/python scripts/eval_full.py run --tag prompted14b --backend prompted --model qwen3:14b
.venv/bin/python scripts/eval_full.py run --tag unified8b  --backend unified  --model vivekananda:8b
.venv/bin/python scripts/eval_full.py compare prompted14b unified8b
```

The harness answers all 150 questions and 50 probes through the real pipeline in auto mode, caches every
answer and judge verdict (resumable), and reports:

- automatic: mode accuracy and confusion, oracle verbatim rate and quoted words, scholar citation coverage,
  persona marker rate, 8-gram overlap of unquoted prose with the corpus (a memorization detector), quote
  guard violations, third-person self-references;
- judged by `qwen3:14b` at temperature 0: first-person consistency, anachronism, generic prose, oracle
  relevance, scholar sentence support against the cited paragraphs, probe agreement with the documented stance.

The judge's first-person verdict proved unreliable (it called "I told the audience at Chicago..." third
person), so the regex count and a human reading are used for that one.

**Run 1 lost to the prompted baseline.** Mode accuracy 0.94 versus 0.973, oracle relevance 0.88 versus
0.989, and the persona marker rate collapsed to 0.579 with 8-gram overlap rising to 7%. It was faster
(7.6 versus 10.1 seconds) and scholar was as good, but it was not the shipping model.

## 13. Diagnosing run 1 and redesigning the data (run 2)

Reading the failures showed one dominant cause. The persona training mix was 899 *verbatim* letters and
dialogues (tiers 2 and 3: real corpus text, first person by construction, no marker, no citation tags)
against only 303 marked synthetic answers (tier 6). The model learned that `persona` mostly means "write in
his letter voice from memory". On stance probes and in-corpus questions it chose persona and produced
unmarked, uncited prose that paraphrased or regurgitated the corpus. That is exactly the failure the whole
cascade exists to prevent: his ideas presented without a verifiable quote.

The second cause was retrieval: the unified path used deterministic retrieval only, while the prompted
baseline expanded queries with the model, so the 8B saw worse passages and picked tangential ones.

The fix the owner chose, run 2:

1. **Tiers 2 and 3 become oracle records.** They are corpus text, so under the hard rule they are
   quotations. Same user turn, but the target is now a selection plan over the answer paragraph and the
   block includes the source paragraph. The verbatim persona register disappears from training entirely.
2. **Tier 1 persona prose is dropped.** Persona now means marked synthetic answers only.
3. **Tier 6 is enlarged** with 124 new topics (work, relationships, politics and identity, technology,
   illness and death, ethics, practice, post-1902 India and the world), 8 questions each, askers varied by
   age, sex, country and faith, to about 1,400 kept records. The old 25% cap is removed.
4. **Tier 9, query expansion**: 1,000 records mapping a question to two or three short search queries, so
   the single 8B can write its own retrieval queries. The serving path now does that expansion by default.
5. Caps so oracle does not swamp the mix: tier 3 at 300, tier 4 at 2,300.

```bash
bash scripts/run_run2_data.sh      # oracle re-render of tiers 2/3, tier 9, enlarged tier 6, assembly (~1.5 h, all on the 14B)
bash scripts/run_run2_train.sh     # train -> export as vivekananda:8b-run2 -> eval -> compare (~8 h)
```

One more bug surfaced in the first run 2 assembly, and it is the kind that would have quietly wasted six
GPU hours: only 3 of the 1,000 tier 9 records survived. The assembler's duplicate-question check works across
tiers, and tier 9 questions are *sampled from the other tiers* on purpose. The fix dedups tier 9 only against
itself. Always read `data/dataset_stats.md` before launching a run; the per-tier table makes this kind of
loss obvious.

Final run 2 mix: 7,311 training records and 384 holdout. Oracle 3,084, persona 1,426 (all marked), scholar
1,382, expand 1,000, chat 803. No record over 8,192 tokens.

## 14. Run 2 results

Training: 914 steps, 6.4 hours, peak 9.4 GiB, masked holdout loss 0.3595 (run 1: 0.588). Exported as
`vivekananda:8b-run2` and `vivekananda:8b-run2-q4`.

| metric | prompted 14B | run 1 8B | run 2 8B |
|---|---|---|---|
| mode accuracy | 0.973 | 0.94 | **0.98** |
| persona marker rate | 1.0 | 0.579 | **1.0** |
| 8-gram overlap with corpus | 1.6% | 7% | 3.5% |
| quote guard violations | 4 | 1 | **0** |
| judged generic prose | 0.095 | 0.20 | 0.115 |
| oracle relevance | **0.989** | 0.88 | 0.919 |
| probe agreement | **0.86** | 0.66 | 0.74 |
| seconds per answer | 10.1 | 7.6 | **6.4** |

Both run 1 regressions are gone. Run 2 routes better than the prompted baseline, never breaks the quote
rule, always marks extrapolation, and is a third faster. It is still behind the 14B on two retrieval-side
measures: in 8 of 99 oracle answers it quoted a true but tangential passage, and it agreed with the
documented stance on fewer probes. The shipping decision between run 2, the prompted cascade, and a third
run aimed at those two numbers rests with the owner; the server default is still the prompted cascade.

## 15. Serving it: API, CLI and PWA

Built before training finished, against the prompted stand-in, so the clients were ready the moment a
model existed. One `answer()` function in `scripts/engine.py` takes a `backend` of `prompted` or `unified`;
the CLI, the server and the eval harness all call it, so they cannot drift.

```bash
.venv/bin/python scripts/server.py --host 0.0.0.0 --port 8765 --backend prompted
.venv/bin/python scripts/ask.py "What is Maya?" --backend unified --model vivekananda:8b-run2
.venv/bin/python scripts/ask.py "..." --show 1          # print the full section behind citation 1
```

- **Server** (`scripts/server.py`, standard library only): `POST /ask` streams server-sent events in the
  order status, mode, token, text, citations, meta, done; `stream: false` returns the same JSON the CLI
  prints with `--json`. Also `/search`, `/section/{id}`, `/paragraph/{id}`, `/health`, and the PWA files.
- **PWA** (`pwa/`, vanilla JS, no build step): one chat screen, mode badge colored per mode, citation chips
  that open the section in a drawer with the quoted paragraph highlighted, history in `localStorage`, a
  service worker so the installed app opens offline. Checked with headless Chrome screenshots, which caught
  two layout bugs.

## 16. Lessons that transfer to the next persona

- **Make the dangerous failure impossible, not improbable.** The oracle never writes text; it points at
  text. The scholar never delivers an uncited sentence. The guard runs on everything. The fine-tune then
  only has to learn to use the contract, not to be trustworthy.
- **Build the prompted version first.** It is the baseline, the data generator, the fallback, and the
  thing that tells you whether the fine-tune is worth shipping. Run 1 would have shipped a worse system
  if there had been nothing to compare it to.
- **Training and serving prompts must be byte-identical**, down to the empty think block. Check it with
  the tokenizer, not by eye.
- **Verbatim corpus text is a quotation, whatever tier it came from.** Putting it under the persona tag
  taught the model to paraphrase the corpus without citations. Re-tiering it as oracle fixed the voice
  problem more than any hyperparameter could have.
- **Read the per-tier dataset table before every run.** Two silent data bugs (the truncated 4K context and
  the tier 9 dedup) were only visible there.
- **Fix the evaluation before the model.** The unmasked holdout loss and the judge's first-person verdict
  both looked like model regressions and were measurement errors.
- **Iterate on data before hyperparameters.** Run 2 changed nothing about the training recipe.
- **Keep the GPU to yourself during training.** A shared Ollama host on the LAN will OOM a run that is
  sized to the card.
