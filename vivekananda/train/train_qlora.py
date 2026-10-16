#!/usr/bin/env python
"""Phase 7: QLoRA fine-tune of Qwen3-8B on data/train.jsonl with Unsloth.

    .venv/bin/python train/train_qlora.py --out train/out/run1            # full run (owner-approved only)
    .venv/bin/python train/train_qlora.py --smoke                         # 2 steps on 4 records, discards output
    .venv/bin/python train/train_qlora.py --out train/out/run1 --resume   # continue from last checkpoint

Targets: the chat-template rendering of each record; loss only on the assistant turn (train_on_responses_only),
which for Qwen3 in non-thinking form is "<think>\n\n</think>\n\n<mode>X</mode>\n..." exactly as Ollama prompts it.
Writes <out>/adapter (LoRA), <out>/checkpoint-*, <out>/metrics.md (loss curves, holdout loss, timing).
"""
import argparse, json, os, pathlib, random, time, datetime
ROOT = pathlib.Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument('--base', default=os.environ.get('VK_BASE', 'Qwen/Qwen3-8B'))
ap.add_argument('--train', default=str(ROOT / 'data/train.jsonl'))
ap.add_argument('--holdout', default=str(ROOT / 'data/holdout.jsonl'))
ap.add_argument('--out', default=str(ROOT / 'train/out/run1'))
ap.add_argument('--epochs', type=float, default=2.0)
ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--rank', type=int, default=32)
ap.add_argument('--alpha', type=int, default=32)
ap.add_argument('--max-seq', type=int, default=8192)
ap.add_argument('--grad-accum', type=int, default=16)
ap.add_argument('--max-steps', type=int, default=-1)
ap.add_argument('--eval-steps', type=int, default=100)
ap.add_argument('--eval-n', type=int, default=100, help='holdout records used for in-training eval loss (full holdout at the end)')
ap.add_argument('--save-steps', type=int, default=100)
ap.add_argument('--seed', type=int, default=3407)
ap.add_argument('--resume', action='store_true')
ap.add_argument('--smoke', action='store_true', help='VRAM/mask smoke test: 4 records, 2 optimizer steps, output to train/out/smoke')
args = ap.parse_args()
if args.smoke:
    args.out = str(ROOT / 'train/out/smoke'); args.max_steps = 2; args.grad_accum = 1; args.eval_n = 4; args.eval_steps = 1; args.save_steps = 1000
out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
t0 = time.time()

from unsloth import FastLanguageModel            # must precede transformers/trl imports
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.base, max_seq_length=args.max_seq, load_in_4bit=True, dtype=None)
model = FastLanguageModel.get_peft_model(
    model, r=args.rank, lora_alpha=args.alpha, lora_dropout=0, bias='none',
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    use_gradient_checkpointing='unsloth', random_state=args.seed)

def load(path, limit=None, shuffle=False):
    rows = [json.loads(l) for l in open(path)]
    if shuffle: random.Random(args.seed).shuffle(rows)
    if limit: rows = rows[:limit]
    texts = [tokenizer.apply_chat_template(r['messages'], tokenize=False, add_generation_prompt=False) for r in rows]
    return Dataset.from_dict({'text': texts, 'id': [r['id'] for r in rows], 'tier': [r['tier'] for r in rows]})

train_ds = load(args.train, limit=4 if args.smoke else None, shuffle=args.smoke)
eval_ds = load(args.holdout, limit=args.eval_n, shuffle=True)
full_holdout = load(args.holdout) if not args.smoke else eval_ds
marker = '<|im_start|>assistant\n<think>\n\n</think>\n\n<mode>'
n_marked = sum(marker in t for t in train_ds['text'])
print(f'records: train {len(train_ds)} eval {len(eval_ds)}; {n_marked} train records carry the mode tag after an empty think block (rest are tier 8 chat)')

cfg = SFTConfig(
    output_dir=str(out), per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
    num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr, lr_scheduler_type='cosine',
    warmup_ratio=0.03, weight_decay=0.01, optim='adamw_8bit', bf16=True, fp16=False,
    logging_steps=5, eval_strategy='steps', eval_steps=args.eval_steps, per_device_eval_batch_size=1,
    save_strategy='steps', save_steps=args.save_steps, save_total_limit=3, seed=args.seed,
    dataset_text_field='text', max_length=args.max_seq, dataset_num_proc=4, packing=False, report_to='none')
trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_ds, eval_dataset=eval_ds, args=cfg)
trainer = train_on_responses_only(trainer, instruction_part='<|im_start|>user\n', response_part='<|im_start|>assistant\n')

# show what is actually trained for one record
ex = trainer.train_dataset[0]
labels = [t for t in ex['labels'] if t != -100]
print('--- trained part of record 0 (first 300 chars):')
print(tokenizer.decode(labels)[:300])
print(f'--- {len(ex["input_ids"])} tokens, {len(labels)} with loss')
assert tokenizer.decode(labels).startswith('<think>'), 'response mask does not start at the think block'

torch.cuda.reset_peak_memory_stats()
res = trainer.train(resume_from_checkpoint=True if args.resume else None)
peak = torch.cuda.max_memory_allocated() / 2**30
final = trainer.evaluate(eval_dataset=full_holdout)
model.save_pretrained(str(out / 'adapter')); tokenizer.save_pretrained(str(out / 'adapter'))

hist = trainer.state.log_history
lines = [f'# Run {out.name}', '', f'base {args.base}, rank {args.rank}, alpha {args.alpha}, lr {args.lr}, epochs {args.epochs}, '
         f'max_seq {args.max_seq}, eff. batch {args.grad_accum}, seed {args.seed}',
         f'train records {len(train_ds)}, steps {trainer.state.global_step}, wall {(time.time()-t0)/3600:.2f} h, '
         f'peak VRAM {peak:.1f} GiB, finished {datetime.datetime.now():%Y-%m-%d %H:%M}', '',
         f'**final holdout loss (all {len(full_holdout)}): {final["eval_loss"]:.4f}**', '',
         '| step | train loss | eval loss | lr |', '|---|---|---|---|']
tl = {}
for h in hist:
    if 'loss' in h: tl[h['step']] = (h['loss'], h.get('learning_rate'))
    if 'eval_loss' in h:
        l, lr = tl.get(h['step'], (None, None))
        lines.append(f'| {h["step"]} | {l if l is None else f"{l:.4f}"} | {h["eval_loss"]:.4f} | {"" if lr is None else f"{lr:.2e}"} |')
lines.append('\ntrain loss every logging step:\n')
lines.append(', '.join(f'{s}:{v[0]:.3f}' for s, v in sorted(tl.items())))
(out / 'metrics.md').write_text('\n'.join(lines) + '\n')
json.dump(hist, open(out / 'log_history.json', 'w'), indent=1)
print('\n'.join(lines[:8]))
