#!/usr/bin/env python
"""Phase 7 pre-check: tokenize train/holdout with the real Qwen3-8B tokenizer and chat template.

Reports token lengths per tier (prompt vs response), records over --max-seq, total trained tokens,
and prints the rendered tail of one record so the think-block / mode-tag layout can be eyeballed.
No GPU, no training.  Usage: .venv/bin/python train/tokencheck.py [--max-seq 8192]
"""
import argparse, json, pathlib, statistics as st, collections, os
ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--base', default=os.environ.get('VK_BASE', 'Qwen/Qwen3-8B'))
ap.add_argument('--max-seq', type=int, default=8192)
args = ap.parse_args()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.base)
RESP = '<|im_start|>assistant\n'

def render(msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

report = []
for split in ('train', 'holdout'):
    rows = [json.loads(l) for l in open(ROOT / 'data' / f'{split}.jsonl')]
    per = collections.defaultdict(list)
    over = []
    total_resp = total_all = 0
    for r in rows:
        text = render(r['messages'])
        assert text.count(RESP) == 1, r['id']
        n_all = len(tok(text, add_special_tokens=False).input_ids)
        n_prompt = len(tok(text.split(RESP)[0] + RESP, add_special_tokens=False).input_ids)
        per[(r['tier'], r['mode'])].append((n_all, n_all - n_prompt))
        total_all += n_all; total_resp += n_all - n_prompt
        if n_all > args.max_seq:
            over.append((r['id'], n_all))
    report.append(f'\n## {split}: {len(rows)} records, {total_all:,} tokens total, {total_resp:,} response tokens, {len(over)} over {args.max_seq}')
    report.append('| tier | mode | n | mean | p90 | max | resp mean | resp max |')
    report.append('|---|---|---|---|---|---|---|---|')
    for (t, m), v in sorted(per.items()):
        a = [x[0] for x in v]; b = [x[1] for x in v]
        a_s = sorted(a)
        report.append(f'| {t} | {m} | {len(v)} | {st.mean(a):.0f} | {a_s[int(0.9*(len(a_s)-1))]} | {max(a)} | {st.mean(b):.0f} | {max(b)} |')
    for i, n in over[:20]:
        report.append(f'- over: {i} {n}')

sample = render(json.loads(open(ROOT / 'data/train.jsonl').readline())['messages'])
report.append('\n## rendered tail of first train record\n```\n' + sample[-700:] + '\n```')
out = '\n'.join(report)
print(out)
(ROOT / 'train/tokencheck.md').write_text('# Token check (' + args.base + ')\n' + out + '\n')
