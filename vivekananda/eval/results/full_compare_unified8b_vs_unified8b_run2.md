| metric | unified8b (unified, vivekananda:8b) | unified8b_run2 (unified, vivekananda:8b-run2) |
|---|---|---|
| n | 200 | 200 |
| mode_accuracy | 0.94 | 0.98 |
| persona_when_oracle_expected | 5 | 0 |
| cascade_rate | 0.0 | 0.0 |
| oracle_n | 75 | 99 |
| oracle_verbatim_rate | 1.0 | 1.0 |
| oracle_mean_quoted_words | 138.76 | 139.404 |
| oracle_over_cap | 0 | 0 |
| oracle_relevant_rate | 0.88 | 0.919 |
| scholar_n | 49 | 50 |
| scholar_citation_coverage | 1.0 | 1.0 |
| scholar_dropped_uncited | 3 | 3 |
| scholar_no_record_rate | 0.02 | 0.06 |
| scholar_support_rate | 0.994 | 0.99 |
| persona_n | 76 | 51 |
| persona_marker_rate | 0.579 | 1.0 |
| persona_cited_sentence_rate | 0.113 | 0.437 |
| persona_8gram_overlap | 0.07 | 0.035 |
| guard_violations | 1 | 0 |
| third_person_refs | 1 | 3 |
| judged_first_person_rate | 0.85 | 0.9 |
| judged_anachronism_rate | 0.0 | 0.005 |
| judged_generic_rate | 0.2 | 0.115 |
| probe_agree_rate | 0.66 | 0.74 |
| mean_seconds | 7.644 | 6.353 |

**unified8b confusion (expected -> predicted):** scholar: {'scholar': 49, 'oracle': 1}; oracle: {'oracle': 45, 'persona': 5}; persona: {'persona': 47, 'oracle': 3}; mixed: {}

**unified8b failure categories:** judge: generic prose (40); persona answer without an extrapolation marker (32); judge: not consistently first person (30); judge: contradicts documented stance (12); judge: oracle quotes not relevant (9); mode persona != expected oracle (5); mode oracle != expected persona (3); judge: N/N scholar sentences unsupported (2); mode oracle != expected scholar (1); N quoted span(s) failed verification (1); N third-person self-reference(s) (1)

**unified8b_run2 confusion (expected -> predicted):** scholar: {'scholar': 50}; oracle: {'oracle': 50}; persona: {'persona': 47, 'oracle': 3}; mixed: {}

**unified8b_run2 failure categories:** judge: generic prose (23); judge: not consistently first person (20); judge: contradicts documented stance (9); judge: oracle quotes not relevant (8); judge: N/N scholar sentences unsupported (3); mode oracle != expected persona (3); N third-person self-reference(s) (3); judge: anachronism (1)
