| metric | unified8b_run2 (unified, vivekananda:8b-run2) | unified8b_run2b (unified, vivekananda:8b-run2) |
|---|---|---|
| n | 200 | 200 |
| mode_accuracy | 0.98 | 0.973 |
| persona_when_oracle_expected | 0 | 1 |
| cascade_rate | 0.0 | 0.0 |
| oracle_n | 99 | 98 |
| oracle_verbatim_rate | 1.0 | 1.0 |
| oracle_mean_quoted_words | 139.404 | 134.52 |
| oracle_over_cap | 0 | 0 |
| oracle_relevant_rate | 0.919 | 0.918 |
| scholar_n | 50 | 50 |
| scholar_citation_coverage | 1.0 | 1.0 |
| scholar_dropped_uncited | 3 | 6 |
| scholar_no_record_rate | 0.06 | 0.06 |
| scholar_support_rate | 0.99 | 0.991 |
| persona_n | 51 | 52 |
| persona_marker_rate | 1.0 | 1.0 |
| persona_cited_sentence_rate | 0.437 | 0.483 |
| persona_8gram_overlap | 0.035 | 0.047 |
| guard_violations | 0 | 0 |
| third_person_refs | 3 | 3 |
| judged_first_person_rate | 0.9 | 0.915 |
| judged_anachronism_rate | 0.005 | 0.005 |
| judged_generic_rate | 0.115 | 0.085 |
| probe_agree_rate | 0.74 | 0.76 |
| mean_seconds | 6.353 | 6.501 |

**unified8b_run2 confusion (expected -> predicted):** scholar: {'scholar': 50}; oracle: {'oracle': 50}; persona: {'persona': 47, 'oracle': 3}; mixed: {}

**unified8b_run2 failure categories:** judge: generic prose (23); judge: not consistently first person (20); judge: contradicts documented stance (9); judge: oracle quotes not relevant (8); judge: N/N scholar sentences unsupported (3); mode oracle != expected persona (3); N third-person self-reference(s) (3); judge: anachronism (1)

**unified8b_run2b confusion (expected -> predicted):** scholar: {'scholar': 50}; oracle: {'oracle': 49, 'persona': 1}; persona: {'persona': 47, 'oracle': 3}; mixed: {}

**unified8b_run2b failure categories:** judge: not consistently first person (17); judge: generic prose (17); judge: oracle quotes not relevant (8); judge: contradicts documented stance (7); judge: N/N scholar sentences unsupported (3); mode oracle != expected persona (3); N third-person self-reference(s) (3); N uncited sentences dropped (1); mode persona != expected oracle (1); judge: anachronism (1)
