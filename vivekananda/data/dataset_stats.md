# Dataset stats

records: train 6195, holdout 327, dropped 551, warnings 28

| tier | mode | train | holdout | mean est. tokens | max est. tokens |
|---|---|---|---|---|---|
| 1 | oracle | 152 | 7 | 2700 | 6153 |
| 1 | persona | 10 | 2 | 3061 | 5964 |
| 2 | persona | 334 | 18 | 2112 | 5341 |
| 3 | persona | 565 | 30 | 2604 | 7038 |
| 4 | oracle | 2775 | 146 | 3370 | 7781 |
| 5 | scholar | 1313 | 69 | 2792 | 7351 |
| 6 | persona | 303 | 16 | 3261 | 6560 |
| 8 | chat | 743 | 39 | 155 | 1126 |

| mode | total |
|---|---|
| oracle | 3080 |
| scholar | 1382 |
| persona | 1278 |
| chat | 782 |

gold paragraph injected into the block (oracle tiers): 464
persona records with the extrapolation marker (tier 6): 319 / 319
records over 8K est. tokens: 0

## Drop reasons

- 411: tierN share cap
- 120: tierN cap
- 7: too long (N chars)
- 4: duplicate question
- 4: tierN no_record cap
- 2: eval question
- 2: bridge contains quotation marks
- 1: persona quoted span not verbatim: 'the whole world requires Light,'

## Warnings (first 20)

- t2-direct-p-010039: warn: guard miss on verbatim text (1): 'I am the Self'
- t2-direct-p-010065: warn: guard miss on verbatim text (1): 'perfected one'
- t2-direct-p-011010: warn: guard miss on verbatim text (1): 'What is Hinduism'
- t2-direct-p-011031: warn: guard miss on verbatim text (2): 'beyond mind and speech'
- t2-direct-p-011057: warn: guard miss on verbatim text (2): 'I am nothing, I am nothing'
- t2-direct-p-011334: warn: guard miss on verbatim text (1): 'Take refuge in Me alone'
- t2-direct-p-011568: warn: guard miss on verbatim text (1): 'Who has come?'
- t2-direct-p-011615: warn: guard miss on verbatim text (2): 'purify the nerves'
- t2-direct-p-011696: warn: guard miss on verbatim text (1): 'Yes, I will,'
- t3-s-0287: warn: guard miss on verbatim text (1): 'Because then'
- t3-s-0337: warn: guard miss on verbatim text (1): 'Wait for a short time.'
- t3-s-0512: warn: guard miss on verbatim text (1): 'Lord have mercy'
- t3-s-0516: warn: guard miss on verbatim text (1): 'this servant'
- t3-s-0552: warn: guard miss on verbatim text (1): 'Ramakrishna, O Lord!'
- t3-s-0555: warn: guard miss on verbatim text (1): 'three cities'
- t3-s-0574: warn: guard miss on verbatim text (1): 'Patriotism is our religion.'
- t3-s-0627: warn: guard miss on verbatim text (1): 'Ramakrishna is God.'
- t3-s-0651: warn: guard miss on verbatim text (1): 'Hand the government of the country over to us'
- t3-s-0790: warn: guard miss on verbatim text (1): 'What is lost in power is gained in speed.'
- t3-s-0796: warn: guard miss on verbatim text (1): 'Hearing Him nothing remains to be heard,\n\nSeeing Him nothing'
