# Token check (Qwen/Qwen3-8B)

## train: 6195 records, 16,900,954 tokens total, 983,729 response tokens, 0 over 8192
| tier | mode | n | mean | p90 | max | resp mean | resp max |
|---|---|---|---|---|---|---|---|
| 1 | oracle | 152 | 2705 | 3616 | 5960 | 63 | 75 |
| 1 | persona | 10 | 2907 | 3283 | 5938 | 105 | 188 |
| 2 | persona | 334 | 2117 | 2958 | 5172 | 212 | 761 |
| 3 | persona | 565 | 2680 | 3677 | 6547 | 405 | 2055 |
| 4 | oracle | 2775 | 3353 | 4562 | 7693 | 65 | 94 |
| 5 | scholar | 1313 | 2914 | 4198 | 7077 | 273 | 613 |
| 6 | persona | 303 | 3287 | 4220 | 6388 | 233 | 345 |
| 8 |  | 743 | 151 | 305 | 906 | 86 | 651 |

## holdout: 327 records, 883,789 tokens total, 53,271 response tokens, 0 over 8192
| tier | mode | n | mean | p90 | max | resp mean | resp max |
|---|---|---|---|---|---|---|---|
| 1 | oracle | 7 | 2582 | 3417 | 3602 | 62 | 68 |
| 1 | persona | 2 | 3681 | 2491 | 4871 | 257 | 445 |
| 2 | persona | 18 | 2291 | 3116 | 4178 | 223 | 618 |
| 3 | persona | 30 | 2588 | 3553 | 4089 | 436 | 2015 |
| 4 | oracle | 146 | 3300 | 4552 | 6641 | 65 | 82 |
| 5 | scholar | 69 | 2886 | 4076 | 5100 | 267 | 478 |
| 6 | persona | 16 | 3291 | 4137 | 4729 | 229 | 261 |
| 8 |  | 39 | 150 | 289 | 328 | 95 | 296 |

## rendered tail of first train record
```
 Vol 7, Notes of Class Talks and Lectures, Comparative Theology that as the world advanced morally and intellectually, differences in religious beliefs faded away, and now there is one all-prevalent doctrine — that of an absolute existence.","ids":["p-012087"]},{"text":"I explained in Vol 2, Jnana-Yoga, Ch. 1: The Necessity of Religion that the evolution of religious ideas has led to the recognition of an absolute existence, which is the foundation of all religions.","ids":["p-001461"]},{"text":"I stated in Vol 1, Lectures and Discourses, What Is Religion? that the highest creed is Oneness, and that the realization of this Oneness is the essence of religion.","ids":["p-000928"]}]}<|im_end|>

```
