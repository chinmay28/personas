| metric | prompted14b (prompted, qwen3:14b) | unified8b_run2 (unified, vivekananda:8b-run2) |
|---|---|---|
| n | 200 | 200 |
| mode_accuracy | 0.973 | 0.98 |
| persona_when_oracle_expected | 1 | 0 |
| cascade_rate | 0.005 | 0.0 |
| oracle_n | 91 | 99 |
| oracle_verbatim_rate | 1.0 | 1.0 |
| oracle_mean_quoted_words | 125.626 | 139.404 |
| oracle_over_cap | 0 | 0 |
| oracle_relevant_rate | 0.989 | 0.919 |
| scholar_n | 56 | 50 |
| scholar_citation_coverage | 1.0 | 1.0 |
| scholar_dropped_uncited | 2 | 3 |
| scholar_no_record_rate | 0.036 | 0.06 |
| scholar_support_rate | 0.989 | 0.99 |
| persona_n | 53 | 51 |
| persona_marker_rate | 1.0 | 1.0 |
| persona_cited_sentence_rate | 0.435 | 0.437 |
| persona_8gram_overlap | 0.016 | 0.035 |
| guard_violations | 4 | 0 |
| third_person_refs | 1 | 3 |
| judged_first_person_rate | 0.91 | 0.9 |
| judged_anachronism_rate | 0.02 | 0.005 |
| judged_generic_rate | 0.095 | 0.115 |
| probe_agree_rate | 0.86 | 0.74 |
| mean_seconds | 10.104 | 6.353 |

**prompted14b confusion (expected -> predicted):** scholar: {'scholar': 50}; oracle: {'oracle': 49, 'persona': 1}; persona: {'persona': 47, 'oracle': 3}; mixed: {}

**prompted14b failure categories:** judge: generic prose (19); judge: not consistently first person (18); judge: contradicts documented stance (7); judge: anachronism (4); judge: N/N scholar sentences unsupported (3); mode oracle != expected persona (3); N quoted span(s) failed verification (3); judge: oracle quotes not relevant (1); mode persona != expected oracle (1); N third-person self-reference(s) (1)

**unified8b_run2 confusion (expected -> predicted):** scholar: {'scholar': 50}; oracle: {'oracle': 50}; persona: {'persona': 47, 'oracle': 3}; mixed: {}

**unified8b_run2 failure categories:** judge: generic prose (23); judge: not consistently first person (20); judge: contradicts documented stance (9); judge: oracle quotes not relevant (8); judge: N/N scholar sentences unsupported (3); mode oracle != expected persona (3); N third-person self-reference(s) (3); judge: anachronism (1)
