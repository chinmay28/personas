# Licensing

| component | license | notes |
|---|---|---|
| code in this repository (scripts, server, PWA, training and eval harness) | AGPL-3.0 (see `LICENSE`) | |
| `*/corpus/` extracted texts | public domain | Swami Vivekananda (1863-1902); the Complete Works editions used are out of copyright. The source e-books are not redistributed. |
| training and evaluation datasets (`*/eval/`, published training sets) | CC0-1.0 | Questions and answers were generated from the public-domain text with local models; no rights are asserted over them. |
| fine-tuned model weights (adapters, merged safetensors, GGUF) | Apache-2.0 | Same license as the Qwen3 base models they derive from. The Ollama Modelfile carries the same notice. |

Each persona directory may add a `LICENSING.md` of its own if its sources differ.
