| metric | prompted14b (prompted, qwen3:14b) |
|---|---|
| n | 200 |
| mode_accuracy | 0.973 |
| persona_when_oracle_expected | 1 |
| cascade_rate | 0.005 |
| oracle_n | 91 |
| oracle_verbatim_rate | 1.0 |
| oracle_mean_quoted_words | 125.626 |
| oracle_over_cap | 0 |
| oracle_relevant_rate | 0.989 |
| scholar_n | 56 |
| scholar_citation_coverage | 1.0 |
| scholar_dropped_uncited | 2 |
| scholar_no_record_rate | 0.036 |
| scholar_support_rate | 0.989 |
| persona_n | 53 |
| persona_marker_rate | 1.0 |
| persona_cited_sentence_rate | 0.435 |
| persona_8gram_overlap | 0.016 |
| guard_violations | 4 |
| third_person_refs | 1 |
| judged_first_person_rate | 0.91 |
| judged_anachronism_rate | 0.02 |
| judged_generic_rate | 0.095 |
| probe_agree_rate | 0.86 |
| mean_seconds | 10.104 |

**prompted14b confusion (expected -> predicted):** scholar: {'scholar': 50}; oracle: {'oracle': 49, 'persona': 1}; persona: {'persona': 47, 'oracle': 3}; mixed: {}

**prompted14b failure categories:** judge: generic prose (19); judge: not consistently first person (18); judge: contradicts documented stance (7); judge: anachronism (4); judge: N/N scholar sentences unsupported (3); mode oracle != expected persona (3); N quoted span(s) failed verification (3); judge: oracle quotes not relevant (1); mode persona != expected oracle (1); N third-person self-reference(s) (1)
