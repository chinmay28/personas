| metric | prompted14b (prompted, qwen3:14b) | unified8b (unified, vivekananda:8b) |
|---|---|---|
| n | 200 | 200 |
| mode_accuracy | 0.973 | 0.94 |
| persona_when_oracle_expected | 1 | 5 |
| cascade_rate | 0.005 | 0.0 |
| oracle_n | 91 | 75 |
| oracle_verbatim_rate | 1.0 | 1.0 |
| oracle_mean_quoted_words | 125.626 | 138.76 |
| oracle_over_cap | 0 | 0 |
| oracle_relevant_rate | 0.989 | 0.88 |
| scholar_n | 56 | 49 |
| scholar_citation_coverage | 1.0 | 1.0 |
| scholar_dropped_uncited | 2 | 3 |
| scholar_no_record_rate | 0.036 | 0.02 |
| scholar_support_rate | 0.989 | 0.994 |
| persona_n | 53 | 76 |
| persona_marker_rate | 1.0 | 0.579 |
| persona_cited_sentence_rate | 0.435 | 0.113 |
| persona_8gram_overlap | 0.016 | 0.07 |
| guard_violations | 4 | 1 |
| third_person_refs | 1 | 1 |
| judged_first_person_rate | 0.91 | 0.85 |
| judged_anachronism_rate | 0.02 | 0.0 |
| judged_generic_rate | 0.095 | 0.2 |
| probe_agree_rate | 0.86 | 0.66 |
| mean_seconds | 10.104 | 7.644 |

**prompted14b confusion (expected -> predicted):** scholar: {'scholar': 50}; oracle: {'oracle': 49, 'persona': 1}; persona: {'persona': 47, 'oracle': 3}; mixed: {}

**prompted14b failure categories:** judge: generic prose (19); judge: not consistently first person (18); judge: contradicts documented stance (7); judge: anachronism (4); judge: N/N scholar sentences unsupported (3); mode oracle != expected persona (3); N quoted span(s) failed verification (3); judge: oracle quotes not relevant (1); mode persona != expected oracle (1); N third-person self-reference(s) (1)

**unified8b confusion (expected -> predicted):** scholar: {'scholar': 49, 'oracle': 1}; oracle: {'oracle': 45, 'persona': 5}; persona: {'persona': 47, 'oracle': 3}; mixed: {}

**unified8b failure categories:** judge: generic prose (40); persona answer without an extrapolation marker (32); judge: not consistently first person (30); judge: contradicts documented stance (12); judge: oracle quotes not relevant (9); mode persona != expected oracle (5); mode oracle != expected persona (3); judge: N/N scholar sentences unsupported (2); mode oracle != expected scholar (1); N quoted span(s) failed verification (1); N third-person self-reference(s) (1)
