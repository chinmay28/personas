# AI Vivekananda: Project Plan

Status: planning complete, no code or data produced yet. Written 2026-10-13, voice decision added same day.
Owner: Chinmay. This plan is the briefing for whichever agent starts the work.
Read it fully before touching anything. Decisions in section 2 are settled; do not reopen them.

## 1. Goal

A local system that answers questions as, and about, Swami Vivekananda using the nine-volume
Complete Works as its only source of his words. It must run on the owner's machine (RTX 4080,
16 GB) through Ollama, shipping as a single fine-tuned 8B model behind a retrieval index,
used through a CLI and an installable PWA served from the same local API.

**Every answer is in the first person, as Vivekananda himself speaking.** The three modes are
grounding levels, not different speakers. They differ in how closely the answer is tied to his
recorded words, and the level is shown to the user. Apply them as a priority cascade, picking
the most grounded mode the question permits:

1. **Scholar.** Questions about his words, positions or life. He recounts what he said and wrote,
   with every claim cited to volume and section. Example: "How did your view of Buddhism
   develop?" Answer shape: "At Chicago I called Buddhism the fulfilment of Hinduism [Vol 1, ...];
   in my later London lectures I ..."
2. **Oracle.** Questions he actually answered. The answer is his own paragraphs, verbatim, with
   minimal first-person bridging text and citations. Example: "What is Karma Yoga?"
3. **Persona.** Questions he never addressed. He reasons from his recorded principles, grounded
   in retrieved passages, and explicitly marks that he is extrapolating. Example: "How should I
   handle burnout at work?"

Mixed questions are split, factual part first as scholar, then advice as persona, visibly
separated. The chosen mode is always displayed to the user.

## 2. Decisions already made

- Cascade order is scholar, then oracle, then persona. Persona is the fallback, never the default.
- All three modes speak in the first person as Vivekananda. No mode describes him in the third
  person. His "now" is his own lifetime; he never refers to his death or to anything after 1902.
- **Hard rule:** anything presented as a quotation must exist verbatim in the corpus. Oracle
  achieves this by construction (span selection, see 5.3). Persona achieves it by a post-check.
- **Oracle quoting policy: key paragraph and cite the rest.** Quote at most one or two paragraphs
  (target under ~250 words of verbatim text per answer), then cite the remaining relevant sections
  by volume, work and title so the reader can go there.
- **Single 8B model ships everything.** Routing and all three modes live in one fine-tuned
  Qwen3-8B exported to GGUF for Ollama. The model emits its chosen mode first, then the answer.
  During development a larger prompted model (qwen3.8:27b) stands in for all of this and produces
  the labeled data to train the 8B on.
- Base: Qwen3-8B QLoRA via Unsloth, non-thinking chat format, as in the sibling books project.
  If a newer ~8B Qwen with Unsloth support is available on Hugging Face, propose it to the owner
  rather than switching silently.
- Refusal and humility policy: no claims about events after 1902. Contemporary political
  questions stop at scholar mode (what he said, cited), no persona extrapolation.
- Checkpoints requiring the owner's go-ahead: (a) after the prompted baseline and eval set exist
  (end of phase 4), (b) before any QLoRA training run starts (phase 7), (c) before any model is
  pushed or shared anywhere. Everything else may proceed without asking.

## 3. Environment and existing assets

Machine (verified 2026-10-13): RTX 4080 16 GB VRAM, 62 GB RAM, ~640 GB free disk, Python 3.12,
NVIDIA driver 580 / CUDA 13. **No PyTorch, uv or conda installed yet.** `~/venv` is an unrelated
FastMCP environment; do not reuse it. Create a fresh environment (uv recommended).

Ollama 0.34 running locally. Models present: qwen3:14b, qwen3.5:9b, qwen3.6:35b, qwen3.8:27b,
qwen35-9b long-context variants. No embedding or reranker model pulled yet. Ollama's OpenAI
endpoint honors `reasoning_effort: "none"` to disable Qwen3 thinking (about 6x faster, quality
comparable for extractive tasks). The 27b model uses ~17 GB so it partially offloads to CPU;
qwen3:14b fits fully on the GPU and is the workhorse for bulk generation.

Existing tooling:
- `datamill` at `~/.local/bin/datamill`, built from `~/src/indtools` (owner's repo,
  github.com/chinmay28/indtools) on uncommitted branch `datamill-reasoning-effort`. It ingests
  documents, chunks them, generates Q&A with Ollama, and exports JSONL, with resumable jobs.
  Its current chunk-to-Q&A mode is NOT the right shape here (it writes model answers; we need
  his passages as the answers). A new mode or sibling tool is expected, see phase 2b.
  Building indtools needs `GOTOOLCHAIN=go1.26.9` because system Go is 1.22 and the shell sets
  GOTOOLCHAIN=local.
- Sibling project: `~/Documents/Books/datasets/` holds a finished 21,489-pair Q&A dataset from
  fiction books, plus `generate.sh` showing the resumable Ollama generation pattern. Fine-tuning
  for that project has not started; nothing there should be modified.
- A children's short-story generator is also planned from `~/Documents/Books/Children's books`.
  It needs the same epub-by-TOC splitter as phase 1 here. Build the splitter reusable.

## 4. Corpus

Source: this directory. One epub (authoritative) and one mobi copy of the same book, plus a
cover image. Public domain text, so no redistribution constraint on derived weights.

Epub facts (inspected 2026-10-13): 752 files, `toc.ncx` with 1,392 entries, about 8.3 million
characters of text (~2 million tokens). All nine volumes present. Structure is clean: every
lecture, book chapter, letter, dialogue and poem is its own TOC entry.

Genres present and how each is used (section 6 has the data tiers):
- Addresses and lectures (Parliament of Religions, London, US, India lectures)
- The four Yoga books (Karma, Raja, Jnana, Bhakti) and Patanjali's Yoga Aphorisms commentary
- Questions and Answers sessions (real audience Q&A, his answers verbatim)
- Conversations and Dialogues (disciple records, mostly third-person "Swamiji said")
- Epistles (several hundred letters, dated and addressed)
- Interviews (newspaper Q&A)
- Writings in prose and poems, translations, notes of class talks, front matter

Known gap: the Complete Works says little about his own life. For biographical questions in
scholar mode, recommend the owner add "The Life of Swami Vivekananda by His Eastern and Western
Disciples" (public domain) to this directory. Do not fabricate biography from memory.

## 5. Architecture

### 5.1 Working directory

Put all generated artifacts in `~/Documents/Books/vivekananda-ai/` (create it), mirroring the
`datasets/` convention of the books project. Suggested layout:

```
vivekananda-ai/
  corpus/      sections.jsonl, paragraphs.jsonl, extraction report
  index/       vector store, BM25 index
  eval/        questions.jsonl, position_probes.jsonl, results/
  data/        training JSONL per tier, merged train/holdout
  train/       Unsloth config, logs, adapter checkpoints
  models/      merged safetensors, GGUF files, Modelfile
  scripts/     everything runnable, each resumable and logged
  README.md    how to run each stage
```

Leave this source directory untouched except for this plan and notes.

### 5.2 Retrieval layer (shared by all modes)

- Two granularities: **sections** (one per TOC entry, for context and citation) and
  **paragraphs** (for verbatim quoting). Paragraph records carry sentence offsets.
- **Hybrid search:** dense vectors plus BM25. Non-negotiable: Sanskrit terms, names and places
  defeat embeddings alone. Fuse with reciprocal rank fusion.
- Embeddings served by Ollama. Candidates to benchmark on the phase 2b recall set:
  `nomic-embed-text`, `bge-m3`, `qwen3-embedding` (whatever tag is current). Pick by recall@5.
- Store: sqlite-vec or LanceDB. No server process. BM25 via SQLite FTS5 or `bm25s`.
- Optional reranker (bge-reranker-v2-m3 class) on the top 30 before passing the top 5 to 8.
  Needs PyTorch; defer until the environment for phase 7 exists unless recall is poor.
- Citation format used everywhere: `[Vol N, Work, Title]`, for example
  `[Vol 1, Karma-Yoga, Ch. 3: The Secret of Work]`. Letters: `[Vol 5, Epistles, To Alasinga, 1894-05-28]`.

### 5.3 The three modes

**Scholar.** Input: question plus top retrieved sections. Output: first-person recounting of
what he said, wrote or did, every factual claim followed by a citation ("As I wrote to Alasinga
in 1894 [Vol 5, Epistles, ...]"). Paraphrase is allowed and is the normal case; quotation marks
are used only around verbatim corpus text, enforced by the quote guard. If the corpus does not
support a claim, he says he has no record of it rather than filling in from the model's
background knowledge. Biographical questions are answered only from what the corpus says.

**Oracle, extractive by construction.** The model does not write the answer text. It sees
numbered retrieved paragraphs with numbered sentences and outputs a selection plan:

```json
{"quotes":[{"para":"p-00412","from":2,"to":5}],
 "bridges":["I take this argument further in the next chapter."],
 "cite_more":["s-0031","s-0032"]}
```

The system renders quotes verbatim from the corpus, inserts bridges as visibly plain text (not
in quotation marks, first person, one sentence each), and appends "See also" citations. Enforce the key-paragraph policy in
the renderer: cap quoted text at ~250 words, move overflow into `cite_more`. If no paragraph
actually answers the question, the oracle returns `{"fallthrough": true}` and the cascade moves
to persona.

**Persona.** Grounded in retrieved passages, opens with an explicit marker when extrapolating
(e.g. "I did not speak to this directly, but..."). Never asserts knowledge of post-1902 events.

**Quote guard, all modes.** Since every mode speaks as him, the guard applies everywhere:
every span inside quotation marks must fuzzy-match a corpus sentence (normalized, similarity
>= 0.95) or be rewritten as unquoted paraphrase. Oracle satisfies it by construction; scholar
and persona by post-check.

### 5.4 Router

Development: prompted classifier (qwen3:14b or 27b, few-shot) returning one of
`scholar | oracle | persona | mixed | decline`, plus retrieval confidence as a feature.
Production: folded into the 8B, which emits `<mode>oracle</mode>` as its first line.
Mode and citations are always shown to the user. A `--mode` override exists for testing.

### 5.5 Inference contract for the single model

The fine-tuned 8B always receives: fixed system prompt, retrieved passages block (numbered
paragraphs and sections, same format as training), then the user question. It emits the mode
tag, then the mode-appropriate output (JSON plan for oracle, prose for scholar and persona).
Train with retrieval context in the prompt so training and inference match exactly.

## 6. Data

### 6.1 Corpus schemas

`sections.jsonl`, one record per TOC entry:
`id, volume, work, title, toc_path (list), genre, date, recipient, place, order, text, char_count`

genre in: `lecture | book_chapter | qa | dialogue | letter | interview | poem | notes |
translation | front_matter | other`

`paragraphs.jsonl`: `id, section_id, para_idx, text, sentences (list of [start,end] offsets)`

### 6.2 Training data tiers (highest fidelity first)

| Tier | Source in corpus | Construction | Mode trained | Notes |
|---|---|---|---|---|
| 1 | Questions and Answers sessions, interviews | Already question/answer pairs in his words | oracle, persona | Highest fidelity, use all |
| 2 | Conversations and Dialogues | Convert third-person reports to first person with an LLM, then verify against source | persona | Keep original alongside for audit |
| 3 | Epistles | LLM synthesizes the implied incoming letter or situation as the user turn; his letter is the reply | persona | Recipient and date as context |
| 4 | Lectures, Yoga books | Reverse-prompt: generate the question each paragraph answers; the paragraph itself is the answer | oracle (as selection plans), retrieval eval | This is the new datamill mode (2b) |
| 5 | Retrieved sections | 27b writes first-person cited recountings ("As I said at ..."); keep only those whose citations verify and pass the quote guard | scholar | Check every citation programmatically |
| 6 | None | Modern questions answered in persona by the 27b using retrieved passages, judged for consistency | persona | Cap at 25% of persona examples |
| 7 | None | Refusals and humility: post-1902 events, politics, identity probes, out-of-domain | all | A few hundred, hand-reviewed |
| 8 | Public general chat data | Small mix to preserve instruction following | n/a | 10 to 15% of total |

Hold out 5% of each tier, stratified, before any training. Deduplicate near-identical passages
(the Complete Works repeats some lectures across volumes).

### 6.3 Training record format

OpenAI-style `messages`: system (fixed), user (retrieval block + question), assistant
(`<mode>X</mode>` newline answer). Loss on assistant tokens only. Sequence length target 8K
(retrieval block dominates). Aim for roughly 8,000 to 15,000 records total.

## 7. Phases

Each phase ends with something usable on its own. Scripts must be resumable and write logs.

**Phase 0: Setup.** Create the working directory, a uv-managed Python 3.12 environment, and
pull one embedding model into Ollama. Confirm `datamill status` still works. Deliverable: README
with environment steps.

**Phase 1: Corpus extraction.** Parse the epub by `toc.ncx`, emit `sections.jsonl` and
`paragraphs.jsonl`, label genre (rule-based from TOC path, LLM-assisted where ambiguous), extract
date and recipient for letters. Acceptance: all 1,392 TOC entries accounted for in an extraction
report; total characters within 5% of 8.3M; genre spot-checked on 50 random sections; date and
recipient present for over 90% of letters. Build the splitter as a reusable component.

**Phase 2a: Index.** Build hybrid index at both granularities. Deliverable: `search` CLI that
prints fused results with citations.

**Phase 2b: Question-for-passage set.** New datamill mode or standalone script: for each
lecture/book paragraph, generate one to two questions it answers using qwen3:14b with reasoning
off, 4 workers, resumable. This doubles as the retrieval eval set (each question has a gold
paragraph). Acceptance: recall@5 >= 0.85 at paragraph level, >= 0.90 at section level on a
1,000-question sample. Tune embedding model, chunking and fusion until met.

**Phase 3: Prompted oracle and scholar.** Implement the span-selection oracle, the renderer with
the key-paragraph cap, the quote guard, and the scholar prompt, all on qwen3.8:27b (fall back to
qwen3:14b if latency is unworkable). Deliverable: `ask` CLI with `--mode`. Acceptance: oracle
verbatim rate 100% by construction, verified by the guard on 100 answers; scholar citations
verify on at least 95% of claims in a 50-answer sample.

**Phase 4: Evaluation set.** 150 questions, 50 per mode, each with expected mode, plus 50
position probes where his stance is documented (Advaita vs dualism, caste, women's education,
Buddhism, ritual, service as worship, renunciation, science and religion) with the supporting
citation. Save under `eval/`. **Checkpoint: owner reviews the set before it is used.**

**Phase 5: Router.** Prompted classifier. Acceptance: >= 90% mode accuracy on the eval set, with
a confusion matrix in the results folder. Persona-when-oracle-was-possible is the error to
minimize; it is the one that lowers grounding.

**Phase 6: Training dataset.** Build tiers 1 to 8 per section 6.2 using the phase 3 system to
produce scholar and oracle targets and the 27b for persona synthesis. Every record carries its
tier and source ids. Run the quote guard and citation checker over the whole set. Deliverable:
`data/train.jsonl`, `data/holdout.jsonl`, and a stats report by tier and mode.

**Phase 7: QLoRA.** **Checkpoint: owner approves before the run starts.** Install PyTorch and
Unsloth in the project environment, download Qwen3-8B weights (~16 GB). Starting hyperparameters:
rank 32, alpha 32, dropout 0, lr 2e-4 cosine, 2 to 3 epochs, batch 1 with gradient accumulation
to effective 16, gradient checkpointing, max sequence 8192. Export adapter, merged safetensors,
GGUF Q8_0 and Q4_K_M, and a Modelfile. `ollama create vivekananda:8b`.

**Phase 8: Evaluate and iterate.** Run the eval set and position probes through the 8B with the
real retrieval pipeline. Compare against the phase 3 prompted system. Deliverable: results table
and a list of failure categories. Iterate on data before touching hyperparameters.

**Phase 9: Ship, CLI and PWA.** Both clients talk to one local HTTP API so they never drift.

- **API server** (local, single process): `POST /ask` streaming the answer over SSE with the mode
  tag first, then text, then citations; `GET /section/{id}` and `GET /paragraph/{id}` returning
  corpus text for citation drill-down; `GET /search?q=` exposing raw hybrid retrieval;
  `GET /health` reporting Ollama and index status. Bind to LAN so a phone can reach it; no auth
  needed on the home network but make the bind address a flag.
- **CLI:** `ask "question"` streams the answer with the mode badge and numbered citations;
  `ask --mode persona` overrides the router; `ask --show 3` prints the full section behind
  citation 3; `search "term"` dumps retrieval results. Also used by the eval scripts, so it must
  have a `--json` output. Build in Go if it is to live in indtools alongside datamill, otherwise
  Python in the project environment; either is fine, pick what the API server is written in.
- **PWA:** static files served by the API server, installable on phone and desktop (web manifest,
  service worker caching the shell, offline message when the server is unreachable). Single
  chat-style screen: question box, streamed answer, mode badge colored per mode, citations as
  tappable chips that open the section text in a drawer, a mode override selector, and
  conversation history kept in the browser (localStorage or IndexedDB), nothing stored server
  side. Keep it dependency-light (vanilla or a small framework, no build step preferred).
- Acceptance: install the PWA on a phone over LAN, ask ten questions across modes, drill into
  five citations, confirm the CLI `--json` output matches the API response for the same question.
- **Checkpoint before anything leaves the machine.**

## 8. Evaluation

- Retrieval: recall@5 and recall@10 on the question-for-passage set.
- Router: accuracy and confusion matrix on the 150-question set.
- Oracle: verbatim rate (must be 100%), relevance judged by the 27b, quoted-word count vs cap.
- Scholar: citation verification rate (does the cited section support the sentence).
- Persona: position-probe agreement with documented stances, anachronism count.
- All modes: quote-guard violations (must be 0 after rewrite), no third-person references to
  himself, judge rubric for voice (consistent first person, period-appropriate register,
  Sanskrit terms used correctly).
- Memorization: n-gram overlap of persona answers with the corpus; high overlap outside quotes
  means regurgitation, investigate epochs and data repetition.
- Human: the owner reads 20 answers per mode at each checkpoint. This is the final judge.

## 9. Risks and policies

- Fabricated quotations are the worst failure. The guard is mandatory in every path.
- Third-to-first-person conversion of dialogues can shift meaning; keep originals and sample-audit.
- Synthetic persona data (tier 6) drifts toward generic inspirational prose; cap it and judge it.
- His views reflect the 1890s; the system reports them faithfully and does not editorialize.
- Contemporary politics: scholar mode only, cited, no extrapolation.
- Long GPU runs: confirm with the owner and log progress so they are resumable.

## 10. Open questions for the owner

1. Add the Eastern and Western Disciples biography for scholar-mode biographical coverage?
2. Should citations be mandatory in every answer, or available on request? Default: mandatory.
3. Any topics to exclude from persona mode entirely beyond contemporary politics?
4. (Settled 2026-10-13) Interface: both a CLI and an installable PWA, see phase 9.

## 11. Related notes

The owner's Claude memory holds notes `vivekananda-ai-project`, `rag-books-plan`,
`books-finetune-dataset` and `user-ml-setup` with the same facts in shorter form. This file is
the authoritative version for this project.
