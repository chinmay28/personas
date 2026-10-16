# AI Vivekananda

Local system that answers as Swami Vivekananda from the nine-volume Complete Works.
The authoritative plan is `PLAN.md` in this directory.
This README documents how to run each stage as it is built.

## Phase 0: environment (done 2026-10-14)

```bash
# uv was installed with the official script into ~/.local/bin
cd personas/vivekananda
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python beautifulsoup4 lxml sqlite-vec numpy requests
ollama pull nomic-embed-text      # first embedding candidate; bge-m3 and qwen3-embedding:0.6b for comparison
```

All scripts are run as `.venv/bin/python scripts/<name>.py`. PyTorch and Unsloth are NOT installed
yet; they come with phase 7 and need the owner's go-ahead.

## Phase 1: corpus extraction (done 2026-10-14)

```bash
.venv/bin/python scripts/extract_corpus.py          # ~15 s, rewrites corpus/
```

- `scripts/epub_split.py` is the reusable epub-by-TOC splitter (library + CLI). It walks the spine
  in document order and assigns every block to the latest `toc.ncx` anchor, so sections may span
  files. Double `<br>` inside one `<p>` is treated as a paragraph break (this epub uses both styles).
  Title lines that repeat the TOC label are dropped. Reuse it for the children's-books project.
- `scripts/extract_corpus.py` applies the Complete Works specifics: genre rules by work title,
  volume by document position, letter metadata (date as ISO, place, recipient, salutation,
  translated_from), sentence offsets, and the editorial endnotes.

Outputs in `corpus/`:

| file | contents |
|---|---|
| `sections.jsonl` | one record per TOC entry (1,391): `id, volume, work, title, toc_path, genre, kind, date, recipient, place, order, text, char_count, ...` |
| `paragraphs.jsonl` | one record per paragraph: `id, section_id, para_idx, text, sentences ([start,end] offsets), note_refs` |
| `endnotes.jsonl` | the book's 430 editorial endnotes (globally numbered; `note_refs` in paragraphs point at `number`) |
| `extraction_report.{md,json}` | acceptance numbers and flagged items |
| `genre_spotcheck.tsv` | 50 random sections with genre, reviewed by hand on 2026-10-14 (all correct) |

Schema notes beyond PLAN.md section 6.1:
- `kind` is `section` (has text), `stub` (under 60 chars, ignore) or `container` (volume/work heading with no text of its own).
- genre set was extended with `essay` (Writings: Prose), `newspaper_report`, and `sayings`; `other` is used for Sister Nivedita's excerpts.
- `WRITINGS: PROSE AND POEMS` entries are labelled poem/essay by a line-length heuristic.
- Inline footnote markers `[n]` are removed from paragraph text and kept in `note_refs`; the 672 inline images (mostly Sanskrit words) are dropped and counted per section.
- Front matter before Volume 1 has `volume: null`.

Acceptance (phase 1): all 1,391 TOC entries accounted for; 8.15M characters (within 5% of the 8.3M estimate; the
difference is the dropped Kindle TOC page, images and footnote markers); letters have a date for 98.6% and a
recipient for 100%; genre spot-check 50/50.

## Phase 2a: hybrid index (done 2026-10-14)

```bash
.venv/bin/python scripts/build_index.py                          # metadata + FTS5 + nomic-embed-text vectors (~70 s)
.venv/bin/python scripts/build_index.py --model bge-m3           # extra vector table per model, same DB
.venv/bin/python scripts/build_index.py --model qwen3-embedding:0.6b
.venv/bin/python scripts/search.py "What is Karma-Yoga?" -k 8            # paragraph level, fused ranks shown
.venv/bin/python scripts/search.py "What is Karma-Yoga?" --level section  # section level
.venv/bin/python scripts/search.py "..." --json                           # machine-readable (used by eval and the API later)
```

- `index/corpus.db` is a single SQLite file: `sections`, `paragraphs`, `endnotes`, FTS5 tables `fts_para`
  and `fts_sec` (unicode61 with diacritics removed, so "Sankhya" matches "Sânkhya"), and one sqlite-vec
  table `vec_para_<model>` per embedding model. Paragraphs shorter than 40 characters are not embedded.
- Dense vectors exist at paragraph level only. Section ranking fuses BM25 over the full section text with
  the best-paragraph dense rank per section (reciprocal rank fusion, k=60). Paragraph ranking fuses
  paragraph dense + paragraph BM25.
- `scripts/vklib.py` holds the shared pieces: DB connection, citation formatter (`[Vol N, Work, Title]`,
  letters as `[Vol N, Epistles, To X, date]`), Ollama embedding client (with per-model query/document
  prefixes), RRF, and the two search functions. Everything later (oracle, scholar, API) imports it.

## Phase 2b: question-for-passage set and retrieval eval (in progress 2026-10-14)

```bash
.venv/bin/python scripts/gen_questions.py --sample 1000     # seeded 1,000-paragraph eval sample, ~30 min on qwen3:14b
.venv/bin/python scripts/gen_questions.py --all             # full run over ~6,500 paragraphs (300-4000 chars), ~3 h; ask the owner first
.venv/bin/python scripts/eval_retrieval.py                  # recall@k per model and method -> eval/results/
```

- Two questions per paragraph: `specific` (uses the passage's own terms) and `general` (plain modern phrasing).
  Output `data/qfp/questions.jsonl`, resumable (skips paragraphs already done), log in `data/qfp/gen.log`.
- Population: paragraphs of 300-4000 chars in genres lecture, book_chapter, essay, notes, qa, interview,
  dialogue, letter, sayings (6,463 paragraphs).

## Phase 3: prompted oracle, scholar, persona (built 2026-10-14, acceptance pending)

```bash
.venv/bin/python scripts/ask.py "What is Karma-Yoga?"                       # auto-routed; prints mode badge, answer, citations
.venv/bin/python scripts/ask.py "..." --mode oracle|scholar|persona           # override the router
.venv/bin/python scripts/ask.py "..." --model qwen3:14b --json                # default model is qwen3:14b (env VK_MODEL)
.venv/bin/python scripts/eval_modes.py --oracle 100 --scholar 50 --model qwen3:14b
```

- `scripts/modes.py`: `retrieve` (hybrid, optionally fused with 2-3 LLM-written keyword queries), `run_oracle`
  (JSON selection plan over numbered sentences, rendered verbatim from the corpus, quoted words capped at 250,
  overflow moved to "I say more on this in ..."), `run_scholar` and `run_persona` (first-person prose with
  `[p-xxxxxx]` tags rendered as numbered citations; tags outside the retrieved set are dropped), `quote_guard`
  (every span in quotation marks must match a corpus sentence at >= 0.95 similarity, else the marks are removed),
  `route` (prompted classifier: scholar | oracle | persona | mixed | decline) and `ask` (the cascade).
- Oracle falls through to persona when the plan is empty; `decline` answers with a fixed refusal.
- All model calls go through Ollama's OpenAI endpoint with reasoning off.

### Phase 2b with datamill (full run completed 2026-10-14)

datamill on `main` (commit e3c30e3 or later) has the question-for-passage mode the plan asked for. The full
run over the 6,463 eligible paragraphs uses it; the 1,000-paragraph sample above came from `gen_questions.py`.

```bash
# input: one JSONL line per paragraph with id (kept as ref), text, and metadata shown to the model
#   (author hint, citation as "source", genre, section_id); built from index/corpus.db
datamill ingest   -project data/qfp/project data/qfp/paragraphs_in.jsonl
datamill generate -project data/qfp/project -provider openai -base-url http://localhost:11434/v1 \
                  -model qwen3:14b -reasoning-effort none -task question -questions 2 -workers 4 -stage questions
datamill export   -project data/qfp/project -format pairs -answer-source chunk -out data/qfp/pairs.jsonl
.venv/bin/python scripts/import_datamill_pairs.py data/qfp/pairs.jsonl       # -> data/qfp/questions_full.jsonl
```

The `author` metadata field tells the model these are Vivekananda's own words, which stops it writing
"the speaker"; without it, datamill's default prompt produced questions like "What does the speaker suggest...".
Progress log: `data/qfp/datamill.log`; `datamill status -project data/qfp/project` shows pending chunks; the run is resumable.

Result: 12,926 questions for all 6,463 eligible paragraphs in `data/qfp/questions_full.jsonl` (the run was interrupted once at
6,410 paragraphs and resumed with the same command).

## Phase 4: evaluation set (drafted 2026-10-14, awaiting owner review)

- `eval/questions.jsonl`: 150 questions, 50 each with `expected_mode` scholar / oracle / persona, plus 20 extras
  (10 decline, 10 mixed, flagged `extra: true`) for the router.
- `eval/position_probes.jsonl`: 50 documented stances (Advaita vs dualism, caste, women, Buddhism, ritual, service,
  renunciation, science, education, Christ, Islam, the Vedas, politics, and others). Each has the stance in one
  sentence, a probe question, and a `support` block with the paragraph id, section id, citation and excerpt that
  documents the stance; every support passage was retrieved from the index and read before being chosen.
- **Checkpoint (2026-10-14):** the owner delegated the review to Claude. Labels were re-read; two vague oracle questions were replaced; stances were cross-checked against public sources (Wikipedia on his teachings) and three probes were re-sourced to stronger corpus passages. Both files are cleared for use.

### Retrieval eval results (2026-10-14, 1,000-paragraph sample, 2,000 questions)

Hybrid paragraph recall of the gold paragraph, `eval/results/retrieval_20261014_160540.json` and `scripts/tune_retrieval.py`:

| variant (hybrid = dense + BM25, RRF) | para@5 | para@10 | para@50 |
|---|---|---|---|
| nomic-embed-text | 0.63 | 0.70 | 0.81 |
| bge-m3 | 0.66 | 0.73 | 0.85 |
| qwen3-embedding:0.6b | 0.63 | 0.71 | - |
| bge-m3/h (header-prefixed paragraphs) | 0.67 | 0.74 | 0.85 |
| **bge-m3/h + bge-m3 + BM25, k=10, cand=100 (default now)** | **0.70** | **0.77** | 0.87 |

- Specific questions reach 0.76@5 / 0.81@10; general questions 0.57 / 0.66. Letters are the weakest genre (0.57@5).
- Header tokens help section-level BM25 (title weight 8) but hurt paragraph-level BM25, so paragraph BM25 is text-only.
- FTS5 note: `bm25()` column weights are positional and include UNINDEXED columns; the first weight is for `id`.
- The plan's acceptance (0.85@5 paragraph, 0.90@5 section) is NOT met. The gold-paragraph metric is pessimistic:
  the Complete Works answer the same general question in many places, so an equally good passage often outranks the
  gold one. Proposed: (a) a cross-encoder reranker over the top 50 once the PyTorch environment exists (ceiling 0.87),
  (b) an LLM-judged "top-5 contains an equivalent passage" metric on a 200-question sample as the real acceptance test.

### Phase 3 acceptance (2026-10-14, qwen3:14b, `eval/results/modes_20261014_164907.json`)

| mode | n | result |
|---|---|---|
| oracle | 100 | verbatim rate 164/164 = 100%, fallthrough 8%, mean 115 quoted words, 0 over the 250 cap, gold section cited 57% |
| scholar | 50 | 65% of sentences carry a citation, 0 invalid tags, 2 of 4 quoted spans failed the guard and were de-quoted, 'no record' 2% |

Oracle meets the acceptance. Scholar does not yet (target: citations verify on 95% of claims); see the fix below.

**Scholar fix (2026-10-14).** The scholar now emits JSON `{"sentences": [{"text", "ids"}]}` instead of free prose; the renderer
drops any sentence whose ids do not resolve to a retrieved passage (strict mode), so every delivered sentence is cited by
construction. On a 20-question re-run: 32 sentences delivered, all cited, 14 model sentences dropped (30%, was 53% with free prose
plus strict dropping), 'no record' 10%. Letter citations now include the place (`[Vol 5, Epistles, To Alasinga, Yokohama, 1893-07-10]`),
which fixed a wrong "I never visited Japan" answer. Whether a cited passage actually supports its sentence is judged by an LLM in phase 8.

## Phase 5: router (2026-10-14)

First run (`eval/results/router_20261014_171419.json`, qwen3:14b, 170 questions incl. extras): accuracy 81.2%.
Persona-when-oracle-expected: 0. Main confusions: 21 oracle questions ("What is Maya?") routed to scholar, 4 scholar to oracle,
3 persona to oracle (acceptable under the cascade preference), and 3 decline questions with post-1902 references routed to scholar
(the dangerous case). The router prompt was then rewritten with explicit definitions and a decline-first rule; second run below.

Second run (`eval/results/router_20261014_172557.json`, rewritten prompt, retrieval evidence without query expansion):
**accuracy 94.1% (160/170)**, meets the >= 90% acceptance. Oracle 50/50, mixed 10/10, scholar 48/50, persona 46/50 (5 sent to
oracle, which the cascade prefers when a passage exists), decline 6/9. Remaining decline misses: "Who should I vote for",
"Ramakrishna Mission's work in the 21st century", "what happened at the Chicago Fair after you left". A deterministic pre-filter
for years > 1902 / "21st century" / election words is the cheap next step.

Third run adds a rule-based pre-filter (`POST1902_RE` in `modes.route`): years after 1902, "21st century", election and
political-office words, Gandhi/Nehru/partition, "after you left/died". Zero false positives on the 150 in-scope questions;
the 20 extras score 20/20 (`eval/results/router_20261014_172750.json`). Combined with run two: 163/170 = 95.9%.

## Phase 6: training dataset (built 2026-10-14)

Scripts: `scripts/traindata.py` (shared: fixed system prompt, retrieval block, record format, oracle-plan builder, letter stripper),
`scripts/build_dataset.py tierN` (one resumable command per tier, log in `data/tiers/build.log`), `scripts/assemble_dataset.py`
(verification, caps, dedupe, stratified 5% holdout, stats). Output: `data/train.jsonl`, `data/holdout.jsonl`, `data/dataset_stats.md`,
per-tier files in `data/tiers/`, drop reasons in `data/tiers/assemble_dropped.jsonl`.

**Record format (plan 6.3).** `messages` = fixed system prompt (`traindata.UNIFIED_SYSTEM`, which defines the three mode tags and their output
formats), user = retrieval block + `QUESTION:`, assistant = `<mode>X</mode>` newline + the mode output. Each record also carries `tier`, `mode`,
`source` (section id, paragraph ids, kind), `block_ids` (paragraph ids in the block), `gold_injected`, and `meta` (rendered text for audit, guard
reports, original question, etc.).

Decisions taken while building (not in the plan):
- **One block format for all modes.** The single model picks the mode after seeing the block, so the block cannot depend on the mode:
  it is always `format_block(ctx, numbered_sentences=True)` (8 hybrid hits + up to 2 section openings), built without LLM query expansion so it is
  deterministic. At inference the API must build the block the same way.
- **Oracle targets are the JSON selection plan**, not the rendered text (the renderer makes the quote verbatim). For tiers 1 and 4 the plan is built
  deterministically from the gold paragraph: the whole paragraph if it is <= 200 words, else the sentence window with the most question overlap;
  the bridge sentence comes from metadata templates (`traindata.bridge_for`); `cite_more` lists retrieved sections of the same work. If retrieval
  missed the gold paragraph it is injected in place of the lowest hit (`gold_injected`, ~30% of tier 1, ~8% of tier 4).
- **Persona tiers exclude their own source section from the block** (tiers 2 and 3), so the model learns voice, not copying; those verbatim answers
  carry no `[p-id]` tags. The extrapolation marker and tags are learned from tier 6, which is the only persona tier with synthetic answers.
- **Scholar targets are the JSON sentence list** with only the sentences whose ids resolve to the block (strict rendering is thus implicit).
- **Mixed mode is not in the training set** (it would need a two-part target); at eval time scholar or persona is accepted for mixed questions.
- **Tier 8 uses a generic system prompt** ("You are a helpful assistant.") and no mode tag, so general instruction following does not collide
  with the decline policy under the Vivekananda prompt. Source: databricks-dolly-15k (CC BY-SA 3.0), sampled by category.
- Models: qwen3:14b for all synthesis (user-turn letters, interview extraction, narrative conversion, second-person rewriting, scholar answers,
  modern questions). qwen3.8:27b measured 13.5 tok/s vs 68 tok/s for the 14b on this GPU, so tier 6 answers also default to the 14b;
  `VK_PERSONA_MODEL=qwen3.8:27b` regenerates them with the 27b (roughly 4 h).

Per tier:
- **Tier 1** (oracle, + a few persona): Q&A sections parsed by the `Q. —`/`A. —` markers (102 pairs); interviews extracted by the 14b and kept only if
  the answer is a verbatim substring of one paragraph (73 pairs; whole-sentence matches become oracle plans, the rest persona prose with the
  paragraph tag). The renderer now strips the `A. —` marker and wrapping quotation marks from quoted spans.
- **Tier 2** (persona): direct-speech dialogues parsed on `Speaker:` lines (`Disciple:` -> `Swamiji:`), answers 150-3,500 chars, verbatim; the 12
  narrative reminiscences are converted to first person by the 14b, kept only when >= 60% of the answer's content words occur in the source,
  it is first person, and the quote guard passes; the source excerpt is stored in `meta.audit_source_excerpt`.
- **Tier 3** (persona): 724 letters (300-9,000 chars, deduplicated by body); header, salutation and signature stripped (`traindata.letter_body`),
  PS kept; the 14b writes the incoming letter that his reply answers (first draft of the prompt produced replies *to* his letter; the prompt now
  says the correspondent has not yet heard from him, and `REPLY_LEAK_RE` rejects leaks).
- **Tier 4** (oracle): one question per paragraph from the 2b set, genre quotas (lecture 1,200, chapter 600, letter 350, notes 300, essay 250,
  dialogue 200, ...), paragraphs with third-person narration ("Swamiji said") excluded, questions rewritten to the second person by the 14b in
  batches of 20 (regex fallback).
- **Tier 5** (scholar): templated questions per section ("What did you write to Mrs. Bull in June 1896?", "What did you say in your lecture ...?")
  plus two 14b-generated "about him" questions for 400 larger sections; answered with the phase 3 scholar prompt; kept when >= 3 sentences survive;
  `no_record` answers capped at 5% at assembly. The scholar prompt was told to name the recipient once and vary openings (the 14b wrote
  "I told X that ..." in every sentence).
- **Tier 6** (persona): 440 modern questions generated by the 14b over 44 topics, answered with the phase 3 persona prompt, quote-guarded, tags
  outside the block removed, capped at 25% of persona records at assembly.
- **Tier 7** (decline): REMOVED at the owner's request (2026-10-14): no refusal examples are trained; the model answers every question in his voice.
- **Tier 8** (chat): Dolly sample, held to 12% of the total at assembly.

Guard fix found while assembling: `QUOTE_RE` paired the closing mark of one one-letter quote with the opening mark of the next (`the little "I"
... the great "I"`) and the guard then stripped the marks; spans must now start and end on a non-space character. `verify_quote` now also does the
corpus lookup for spans >= 10 characters (was > 20), so short quoted phrases in verbatim text verify.

**Ollama context fix (2026-10-14).** The Ollama service runs model slots with a 4,096-token context by default and silently truncates longer
prompts (`truncated = 1` in `journalctl -u ollama`); retrieval blocks with 10 passages reach 3,300 tokens at the 90th percentile and 6,200 at
most, so about 5% of phase 3/5 generation prompts were being cut (10 truncations were logged during the tier 5/6 samples). `modes.chat` now
calls the native `/api/chat` endpoint with `num_ctx` = 8192 (`VK_NUM_CTX`), which the OpenAI-compatible endpoint cannot set per request. The
phase 3 and 5 eval numbers above were measured before this fix; phase 8 re-runs them.

**No refusals anywhere (owner decision, 2026-10-14, "nothing is banned").** Removed: the decline tier (7), the decline mode from the
training prompt, the router's decline label, the `POST1902_RE` pre-filter and decline branch in `modes.route`/`modes.ask`, the "know nothing
after 1902 / never mention your death" sentences in `PERSONA_PREAMBLE`, the "no claims after 1902" line in the persona prompt, and the
post-1902 filters on synthesized letters, modern questions and persona answers. Later events, politics and identity questions now route to
persona (the router prompt lists them there). The 10 decline extras in `eval/questions.jsonl` keep their label for the record; `eval_router.py`
will count them as misses until the eval set is relabeled (persona is the expected route now).

## Phase 7: QLoRA (prepared 2026-10-15, training run awaits owner approval)

Everything in `train/`:
- `tokencheck.py`: real Qwen3-8B tokenizer lengths for train/holdout per tier (prompt vs response), records over 8,192, total trained tokens,
  and the rendered tail of one record. Writes `train/tokencheck.md`. No GPU.
- `train_qlora.py`: Unsloth QLoRA per the plan (4-bit base, rank 32, alpha 32, dropout 0, lr 2e-4 cosine, 3% warmup, 2 epochs, batch 1 x
  grad-accum 16, max_seq 8192, unsloth gradient checkpointing, adamw_8bit). Loss is on the assistant turn only
  (`train_on_responses_only`), i.e. on `<think>\n\n</think>\n\n<mode>X</mode>\n...` which is byte-for-byte what Ollama puts after
  `<|im_start|>assistant` when thinking is off, so training and serving prompts match. Eval loss on 100 holdout records every 100 steps,
  full holdout at the end; writes `<out>/adapter`, checkpoints, `metrics.md`, `log_history.json`. `--smoke` = 4 records, 2 steps, to
  check VRAM and the label mask; `--resume` continues from the last checkpoint.
- `export.py`: merge adapter -> bf16 safetensors, `convert_hf_to_gguf.py` (llama.cpp at `~/src/llama.cpp`, CPU build of
  `llama-quantize` only) -> f16, Q8_0 and Q4_K_M GGUFs in `models/`, Modelfiles using the Qwen3 template copied from Ollama's own
  qwen3:14b (`train/qwen3.template`) with num_ctx 8192 and temperature 0.3, then `ollama create vivekananda:8b` (Q8_0) and `vivekananda:8b-q4`.
- Base weights: `Qwen/Qwen3-8B` (post-trained hybrid-thinking model, bf16, 16 GB) in the Hugging Face cache; the instruct model rather
  than the base so tier 8 chat and JSON discipline start from a model that already follows instructions. Non-thinking format throughout.
- Environment: unsloth + torch (CUDA wheels) + trl + peft + bitsandbytes in `.venv` (`train/install.log`).

Expected cost: ~6,200 records x 2 epochs / 16 = ~775 optimizer steps; at ~3k tokens per record this is roughly 4 h on the 4080
(to be confirmed by the smoke test's step time).

**Smoke test (2026-10-15, `--smoke`, `train/smoke.log`):** model loads in 4-bit, the label mask starts exactly at `<think>` and covers only the
assistant turn (3,256-token oracle record -> 52 loss tokens), 2 optimizer steps ran, peak VRAM **8.5 GiB**, **~3.3 s per ~3k-token record**.
Real-tokenizer sizes (`train/tokencheck.md`): train 16.9M tokens per epoch, 0 records over 8,192, only 6% of tokens are response tokens.
Corrected cost: 6,195 records x 2 epochs x ~2.8 s = **~9.5-10 h** for the planned 2 epochs (about 5 h per epoch), not the 4 h guessed above.
Scheduling note: Ollama is also used from the LAN (another client loaded a 27B model during the smoke test and took 13.4 GB of the
GPU, which made the first attempt OOM). The training run needs the GPU to itself; a concurrent large-model Ollama load could OOM it mid-run
(checkpoints every 100 steps + `--resume` limit the loss).

## Phase 9 (built 2026-10-15, ahead of the phase 7 run): API server, CLI, PWA; plus the phase 8 harness

Built against the prompted stand-in so the clients and the evaluation are ready the moment the 8B exists. Nothing here needs the
trained model; a `--backend` switch picks the inference path.

**Backends (`scripts/engine.py`, one `answer()` used by the CLI, the server and the eval).**
- `prompted`: the phase 3 cascade on a stock Ollama model (router, per-mode prompts, LLM query expansion), `modes.ask`.
- `unified` (`scripts/unified.py`): the fine-tuned single model. The prompt is byte-for-byte the phase 6 training prompt
  (`UNIFIED_SYSTEM`, numbered-sentence block from deterministic retrieval, question) rendered through the Qwen3 non-thinking template
  (checked equal to `tokenizer.apply_chat_template(..., enable_thinking=False)`), sent to Ollama's `/api/generate` in raw mode. Raw
  mode is used because the chat endpoint cannot prefill an assistant turn with the Qwen3 template; a `--mode` override is a prefilled
  `<mode>X</mode>` line. The model's output goes through the same renderers as the prompted system (`oracle_from_plan`,
  `scholar_from_plan`, `persona_from_text`, now factored out of `run_*` in `modes.py`), so the quote guard and strict citations apply
  identically. Oracle fallthrough cascades to a prefilled persona generation. `mixed` is not a trained mode: two prefilled generations.
  Default model `vivekananda:8b` (`VK_UNIFIED_MODEL`); tested for plumbing with `--model qwen3:14b` as a stand-in (it follows the
  format well enough to exercise every path).

**API server (`scripts/server.py`, stdlib only, one process).** `python scripts/server.py --host 0.0.0.0 --port 8765 --backend prompted`.
`POST /ask` streams SSE events in this order: `status`, `mode` (as soon as it is known, before generation), `token` (persona prose as it
is generated, unified backend only), `text`, `citations`, `meta`, `done` (or `error`). `"stream": false` or `GET /ask?q=&stream=0` returns
the full result JSON, which is exactly what the CLI prints with `--json` (verified: same keys, same shape; the text differs run to run at
temperature 0.3). `GET /search?q=` raw hybrid retrieval, `GET /section/{id}` and `GET /paragraph/{id}` for citation drill-down,
`GET /health` (Ollama reachability, models present, index counts, active backend/model). Static files from `pwa/` at `/`. Binds to the LAN
by default; no auth (home network, per the plan). Path traversal outside `pwa/` returns 404.

**CLI (`scripts/ask.py`).** `--backend prompted|unified`, `--mode`, `--model`, `--json`, `--show N` (prints the full section behind
citation N with the quoted paragraph marked), `--api http://host:8765` (answers through a running server instead of in-process; `--json`
output is then the server's JSON). The persona answer streams token by token on the unified backend. `scripts/search.py` is unchanged.

**PWA (`pwa/`, vanilla JS, no build step).** Single chat screen: question box, mode override select, streamed answer with a colored mode
badge (oracle amber, scholar blue, persona violet, mixed teal), inline `[n]` markers and citation chips that open the section in a drawer
(bottom sheet on phones, side panel on wide screens) with the cited paragraph highlighted, "see also" chips, settings drawer (backend and
model override, retrieval/timing details, clear/export history), history in `localStorage` (last 200 exchanges, nothing server side),
web manifest + icons (`pwa/icon-*.png`, generated with Pillow), service worker caching the shell so the installed app opens offline and
then reports the server as unreachable. API calls are never cached. `/?q=...&mode=...` asks immediately. Light and dark themes.
Rendered in headless Chrome to check the layout (a `[hidden]` vs `display:flex` bug was caught that way); **installing on a phone and
the ten-question / five-citation acceptance walk-through are still to be done by the owner.**

**Phase 8 harness (`scripts/eval_full.py`).** `run --tag T --backend B --model M [--judge qwen3:14b] [--limit N] [--only questions|probes]`
answers `eval/questions.jsonl` (150; `--include-extras` adds the 20) and `eval/position_probes.jsonl` (50) in auto mode through the real
pipeline, caches answers and judge verdicts per question (resumable), and writes `eval/results/full_<tag>.json` + `.md`;
`compare T1 T2 ...` prints the metrics side by side. Automatic checks: mode accuracy and confusion, persona-when-oracle-expected,
cascade rate, oracle verbatim rate / quoted words / over-cap, scholar citation coverage / dropped sentences / no-record rate, persona
extrapolation-marker rate, 8-gram overlap of unquoted prose with the corpus (memorization), quote-guard violations, third-person
self-references. LLM judge (qwen3:14b, JSON, temperature 0): first-person consistency, anachronism, generic prose (all answers), oracle
relevance, scholar sentence support against the cited paragraphs, probe agreement with the documented stance. Failure categories are
listed per run. Fix made while building it: `modes.split_sentences` now ends a sentence at a citation marker (`...untrue.[1] They`),
which the corpus splitter did not know; scholar sentence counts and the eval's per-sentence support check were wrong without it.

**Baseline run started 2026-10-15 09:30:** `eval_full.py run --tag prompted14b --backend prompted --model qwen3:14b` (200 items, ~10 s per
answer, then judging), log in `eval/results/full_prompted14b.log`. This is the phase 3 system the 8B must beat.

**Browser walk-through (2026-10-15, `scripts/pwa_smoke.py`, headless Chrome over the DevTools protocol):** `/?q=What is the ideal of
womanhood?` produced an oracle card in 16 s with the badge, two inline markers, two citation chips and a "see also" chip; tapping the first
chip opened the drawer on `[Vol 8, Lectures and Discourses, Women of India]` (46 paragraphs) with the quoted paragraph highlighted and scrolled
into view; the exchange was saved to `localStorage`; the service worker was active. Two PWA bugs were found and fixed by screenshots
(`[hidden]` lost to the drawer's `display:flex`; the app's `history` array shadowed `window.history`). Content note from the same run: the
prompted oracle (14b) quoted a third-person newspaper report ("The speaker closed with...") as his own words. Tier 4 of the training set
excludes such paragraphs, so the 8B should not; the harness counts it under third-person references only when it is outside quotation
marks, so a judged "first_person" miss is the signal to watch.

### Phase 8 baseline: prompted system, qwen3:14b (2026-10-15, `eval/results/full_prompted14b.{json,md}`)

200 items (150 questions + 50 probes), auto mode, real retrieval with query expansion, 10.1 s per answer.

| metric | value | note |
|---|---|---|
| mode accuracy (150) | 0.973 | 1 persona-when-oracle, 3 oracle-when-persona |
| oracle verbatim rate | 1.000 (91 answers) | mean 126 quoted words, 0 over cap, judged relevant 0.989 |
| scholar citation coverage | 1.000 (56) | judged support 0.989, no-record 3.6%, 2 uncited sentences dropped in total |
| persona marker rate | 1.000 (53) | 43.5% of persona sentences cite a passage; 8-gram overlap with corpus 1.6% (no regurgitation) |
| quote-guard violations | 4 | all persona; the spans were unquoted by the guard (paraphrase), none reached the user as quotes |
| third-person self-references (regex) | 1 | probe-50 |
| probe agreement (judge) | 0.86 | 7 "disagree": mostly the oracle quoting a true but different passage (probe-17 renunciation vs householder, probe-40 Purusha vs freedom) |
| judge: anachronism / generic | 0.02 / 0.095 | generic hits are persona answers on modern questions |
| judge: first person | 0.91 | **not trustworthy**: the 14b judge called "I told the audience at Chicago..." third person in 18 cases; use the regex count and the owner's reading instead |

Failure categories to watch when the 8B is compared: oracle choosing a tangential passage on stance questions (retrieval/selection, not voice),
generic persona prose on modern questions, and quoted spans invented by persona (caught by the guard, but the 8B should not produce them).
The 8B run: `.venv/bin/python scripts/eval_full.py run --tag unified8b --backend unified` then `eval_full.py compare prompted14b unified8b`.

### Phase 7 run1 started (2026-10-15 21:10 PDT, owner-approved)

`.venv/bin/python train/train_qlora.py --out train/out/run1` (2 epochs, 776 optimizer steps, eff. batch 16, rank 32, lr 2e-4, max_seq 8192),
launched detached with `setsid nohup`, log `train/run1.log`. Measured ~27 s/step at 10.1 GiB VRAM, so about 5.8 h wall (expected to finish ~03:00 on 2026-10-16) (the smoke-test
estimate of 9.5-10 h was pessimistic). Checkpoints every 100 steps under `train/out/run1/checkpoint-*`; `--resume` continues from the last one
if the run is interrupted. Next after completion: `train/export.py` (merge, GGUF Q8_0 + Q4_K_M, `ollama create vivekananda:8b`), then
`scripts/eval_full.py run --tag unified8b --backend unified` and `compare prompted14b unified8b`.

### Phase 7 run1 finished (2026-10-16 03:10, `train/out/run1/`; loss table kept as `train/run1_metrics.md`)

776 steps in 6.0 h (20.6 s/step once warm, peak VRAM 9.4 GiB), no interruption. Train loss fell from ~2.0 to ~0.9 (noisy per-step, eff. batch 16);
masked eval loss on the 100-record holdout subset 0.641 (step 100) -> 0.588 (step 776), still falling slowly, so 2 epochs did not overfit.
`metrics.md` also prints "final holdout loss (all 327): 2.3517". That number is an artifact, not a regression: the raw full holdout was handed
to `trainer.evaluate`, which tokenizes it without the response-only mask and scores every token including the ~4,000-token retrieval
prompt (an oracle record has ~60 loss tokens out of ~4,000). `train_qlora.py` now tokenizes and masks the full holdout before the final
evaluate (checked on 8 records: every loss region starts at `<think>`); the 0.588 figure is the one to use for run1. Export started
2026-10-16 ~08:00 (`train/export.py`, log `train/export.log`); Unsloth downloads `unsloth/Qwen3-8B` (15 GB) for the 16-bit merge even
though `Qwen/Qwen3-8B` is cached, because the adapter's base is `unsloth/Qwen3-8B-unsloth-bnb-4bit`.

Export finished 2026-10-16 08:11 (13 min including the download): `models/vivekananda-8b-{f16,Q8_0,Q4_K_M}.gguf`, Ollama models
`vivekananda:8b` (Q8_0, 8.7 GB) and `vivekananda:8b-q4` (5.0 GB). Smoke test through `ask.py --backend unified`: oracle quoted a verified
passage (8.7 s), scholar cited 10/10 sentences (7.2 s), persona marked its extrapolation (3.5 s).

### Phase 8: unified 8B vs prompted 14b (2026-10-16, `eval/results/full_unified8b.{json,md}`, `full_compare_prompted14b_vs_unified8b.md`)

`eval_full.py run --tag unified8b --backend unified` (200 items, 7.6 s per answer vs 10.1 s for the prompted cascade, no LLM query expansion).
**run1 does not beat the baseline; it is not the shipping model.** Side by side (prompted14b -> unified8b):

| metric | prompted14b | unified8b | reading |
|---|---|---|---|
| mode accuracy (150) | 0.973 | 0.940 | 5 oracle questions went to persona (vs 1), 3 persona to oracle, 1 scholar to oracle |
| oracle verbatim rate | 1.000 | 1.000 | 75 oracle answers, mean 139 quoted words, 0 over cap |
| oracle judged relevant | 0.989 | 0.880 | 9 tangential passages (soul immortal, Om, Chitta/Vrittis, ...); no query expansion on this path |
| scholar coverage / support | 1.0 / 0.989 | 1.0 / 0.994 | 49 answers; scholar is as good as the baseline |
| persona marker rate | 1.000 | 0.579 | on persona-expected questions 44/47 are marked; the 32 unmarked answers are 24 probes + 5 oracle questions the model routed to persona |
| persona 8-gram overlap | 0.016 | 0.070 | some corpus regurgitation in the unmarked answers ("a terrible engine that will destroy everything") |
| quote-guard violations | 4 | 1 | |
| judged first person / generic | 0.91 / 0.095 | 0.85 / 0.20 | 14b judge, unreliable on first person; the generic verdicts cluster on the unmarked persona answers |
| probe agreement | 0.86 | 0.66 | probes routed oracle 26 / persona 24 (baseline 39 / 6 / 5 scholar); oracle agree 16/26, persona 17/24 |

Diagnosis. The persona tier mix is 899 verbatim letters and dialogues (tiers 2 and 3: no marker, no `[p-id]` tags, first person
by construction) against 303 marked synthetic answers (tier 6). The model learned that `persona` mostly means "write in his letter voice
from memory": for position probes and in-corpus questions it emits `<mode>persona</mode>` and produces unmarked, uncited prose that
paraphrases or regurgitates the corpus, which is the exact failure the cascade exists to prevent (his ideas presented without a verifiable
quote). On questions that are genuinely outside the corpus (tier-6-like) the learned behaviour is fine (44/47 marked, cited passages).
The second regression, oracle relevance, is retrieval side: the unified prompt uses deterministic hybrid retrieval only, and the 8B picks a
tangential passage more often than the 14b with expanded queries. Several probe "disagree" verdicts are the judge being wrong
(probe-28 "education is the manifestation of the perfection already in man" is the documented stance), so the 0.66 is a floor, not the truth.

Options for run2 (owner decision; another ~6 h GPU run plus re-eval):
1. Re-tier the verbatim persona data: tiers 2 and 3 are corpus text, so render them as oracle records (quoted verbatim with the citation)
   or drop them, and train persona only on marked, quote-guarded synthetic answers (tier 6, possibly enlarged to ~1,000 with the 14b).
   This matches the hard rule (anything verbatim is a quote) and removes the unmarked register entirely.
2. Keep tiers 2 and 3 for voice but under a fourth tag (`<mode>voice</mode>`) that the renderer treats like persona with a mandatory
   marker; more complex, keeps the letter register as a style signal.
3. Add LLM query expansion to the unified path (or retrieve with the 8B's own rewritten query) to recover oracle relevance.
Until then the prompted cascade on qwen3:14b (`--backend prompted`, the server default) remains the system of record.
