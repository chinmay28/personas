# personas

AI models and RAGs for historical figures. Each persona is a local system that answers in the
figure's own voice, grounded in their actual writings, and runs on a single consumer GPU through
Ollama. The hard rule shared by every persona: anything presented as a quotation must exist
verbatim in the source corpus.

| persona | corpus | status | directory |
|---|---|---|---|
| Swami Vivekananda | Complete Works, 9 volumes (1,391 sections, 20,694 paragraphs) | retrieval index, prompted modes, eval harness, API/CLI/PWA done; 8B QLoRA in progress | [`vivekananda/`](vivekananda/) |

## How a persona is built

1. **Corpus**: the public-domain works are split by table of contents into sections and paragraphs with metadata (`corpus/`).
2. **Index**: SQLite with FTS5 and sqlite-vec vectors from an Ollama embedding model, fused with reciprocal rank fusion (`scripts/build_index.py`).
3. **Modes**: three grounding levels, all in the first person. *scholar* recounts what the figure said with citations, *oracle* answers with verbatim passages, *persona* extrapolates from the figure's principles only when the corpus is silent, and says so. A router picks the mode; a quote guard rejects any quotation that is not in the corpus.
4. **Fine-tune**: a QLoRA on Qwen3-8B learns the voice, the mode tag and the citation format from a dataset generated out of the corpus (`train/`).
5. **Eval**: automatic checks (verbatim rate, citation coverage, mode accuracy, memorization) plus an LLM judge and a set of stance probes (`eval/`).
6. **Clients**: one local HTTP server with server-sent events, a CLI, and an installable PWA (`scripts/server.py`, `scripts/ask.py`, `pwa/`).

The per-persona README is the runbook; `PLAN.md` in each directory records the design decisions.

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com), and an NVIDIA GPU
with 16 GB for training (inference of the 8B models fits in much less).

```bash
git clone https://github.com/chinmay28/personas.git && cd personas
uv sync                       # retrieval, server, CLI, eval
uv sync --extra train         # adds PyTorch (CUDA), Unsloth, TRL for the QLoRA and GGUF export
ollama pull bge-m3            # embedding model used by the default retrieval
```

Then follow the README in the persona's directory. What is **not** in git, by design: the vector
index (rebuilt from `corpus/` in about a minute), generated training sets, model weights and GGUF
files (published separately, see each persona's README), and the source e-books.

Licensing: code is AGPL-3.0, corpora are public domain, datasets CC0, weights Apache-2.0. See
[`LICENSING.md`](LICENSING.md).
