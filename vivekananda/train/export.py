#!/usr/bin/env python
"""Phase 7 export: merge the LoRA adapter into Qwen3-8B (bf16 safetensors), convert to GGUF, quantize, register in Ollama.

    .venv/bin/python train/export.py --run train/out/run1 [--name vivekananda:8b] [--gguf-prefix vivekananda-8b] [--skip-ollama]

Produces  <run>/merged/ (safetensors + tokenizer), models/<prefix>-f16.gguf, -Q8_0.gguf, -Q4_K_M.gguf, models/Modelfile (for the default name),
and `ollama create <name>` (Q8_0) plus <name>-q4 (Q4_K_M). Needs llama.cpp built (build/bin/llama-quantize) at ~/src/llama.cpp or $VK_LLAMA_CPP and the `gguf` package.
"""
import argparse, os, pathlib, subprocess, sys, shutil
ROOT = pathlib.Path(__file__).resolve().parents[1]
LLAMA = pathlib.Path(os.environ.get('VK_LLAMA_CPP', pathlib.Path.home() / 'src/llama.cpp'))
ap = argparse.ArgumentParser()
ap.add_argument('--run', default=str(ROOT / 'train/out/run1'))
ap.add_argument('--base', default=os.environ.get('VK_BASE', 'Qwen/Qwen3-8B'))
ap.add_argument('--name', default='vivekananda:8b')
ap.add_argument('--gguf-prefix', default='vivekananda-8b', help='file stem under models/ (use e.g. vivekananda-8b-run2 to keep an earlier export)')
ap.add_argument('--skip-ollama', action='store_true')
ap.add_argument('--skip-merge', action='store_true')
args = ap.parse_args()
run = pathlib.Path(args.run); merged = run / 'merged'; models = ROOT / 'models'; models.mkdir(exist_ok=True)

def sh(*cmd):
    print('+', ' '.join(map(str, cmd)), flush=True); subprocess.run(list(map(str, cmd)), check=True)

if not args.skip_merge:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=str(run / 'adapter'), max_seq_length=8192, load_in_4bit=True, dtype=None)
    model.save_pretrained_merged(str(merged), tokenizer, save_method='merged_16bit')

f16 = models / f'{args.gguf_prefix}-f16.gguf'
sh(sys.executable, LLAMA / 'convert_hf_to_gguf.py', merged, '--outfile', f16, '--outtype', 'f16')
quant = LLAMA / 'build/bin/llama-quantize'
outs = {}
for q in ('Q8_0', 'Q4_K_M'):
    outs[q] = models / f'{args.gguf_prefix}-{q}.gguf'
    sh(quant, f16, outs[q], q)

template = (ROOT / 'train/qwen3.template').read_text()
for q, name in ((('Q8_0'), args.name), ('Q4_K_M', args.name + '-q4')):
    mf = models / (f'Modelfile.{q}' if args.gguf_prefix == 'vivekananda-8b' else f'Modelfile.{args.gguf_prefix}.{q}')
    mf.write_text(f'FROM ./{outs[q].name}\n{template}\nPARAMETER num_ctx 8192\nPARAMETER temperature 0.3\nPARAMETER top_p 0.9\n'
                  'PARAMETER top_k 20\nPARAMETER repeat_penalty 1\nPARAMETER stop <|im_start|>\nPARAMETER stop <|im_end|>\n'
                  'LICENSE "Apache-2.0 (Qwen3-8B base); fine-tuned on the public-domain Complete Works of Swami Vivekananda"\n')
    if not args.skip_ollama:
        sh('ollama', 'create', name, '-f', mf)
if args.gguf_prefix == 'vivekananda-8b':
    shutil.copy(models / 'Modelfile.Q8_0', models / 'Modelfile')
print('done:', *outs.values(), sep='\n  ')
