# Dataset stats

records: train 7311, holdout 384, dropped 1342, warnings 1

| tier | mode | train | holdout | mean est. tokens | max est. tokens |
|---|---|---|---|---|---|
| 1 | oracle | 151 | 8 | 2700 | 6153 |
| 2 | oracle | 309 | 16 | 2319 | 5663 |
| 3 | oracle | 285 | 15 | 2430 | 5293 |
| 4 | oracle | 2185 | 115 | 3384 | 7781 |
| 5 | scholar | 1313 | 69 | 2791 | 7351 |
| 6 | persona | 1355 | 71 | 3339 | 7028 |
| 8 | chat | 763 | 40 | 160 | 1034 |
| 9 | expand | 950 | 50 | 184 | 241 |

| mode | total |
|---|---|
| oracle | 3084 |
| persona | 1426 |
| scholar | 1382 |
| expand | 1000 |
| chat | 803 |

gold paragraph injected into the block (oracle tiers): 685
persona records with the extrapolation marker (tier 6): 1426 / 1426
records over 8K est. tokens: 0

## Drop reasons

- 915: tierN cap
- 390: tierN share cap
- 12: tierN verbatim persona (runN)
- 7: too long (N chars)
- 4: duplicate question
- 4: tierN no_record cap
- 2: eval question
- 2: bridge contains quotation marks
- 1: persona quoted span not verbatim: 'the whole world requires Light,'
- 1: persona quoted span not verbatim: 'I reject the Vedas!'
- 1: persona quoted span not verbatim: 'walking corpse'
- 1: persona quoted span not verbatim: 'the will is stronger than anything else'
- 1: persona quoted span not verbatim: 'differentiation is the life of thought'
- 1: persona quoted span not verbatim: 'money is nothing'

## Warnings (first 20)

- t4-p-003974: warn: quoted words 306
