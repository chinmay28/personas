| metric | unified8b_run2b (unified, vivekananda:8b-run2) |
|---|---|
| n | 200 |
| mode_accuracy | 0.973 |
| persona_when_oracle_expected | 1 |
| cascade_rate | 0.0 |
| oracle_n | 98 |
| oracle_verbatim_rate | 1.0 |
| oracle_mean_quoted_words | 134.52 |
| oracle_over_cap | 0 |
| oracle_relevant_rate | 0.918 |
| scholar_n | 50 |
| scholar_citation_coverage | 1.0 |
| scholar_dropped_uncited | 6 |
| scholar_no_record_rate | 0.06 |
| scholar_support_rate | 0.991 |
| persona_n | 52 |
| persona_marker_rate | 1.0 |
| persona_cited_sentence_rate | 0.483 |
| persona_8gram_overlap | 0.047 |
| guard_violations | 0 |
| third_person_refs | 3 |
| judged_first_person_rate | 0.915 |
| judged_anachronism_rate | 0.005 |
| judged_generic_rate | 0.085 |
| probe_agree_rate | 0.76 |
| mean_seconds | 6.501 |

**unified8b_run2b confusion (expected -> predicted):** scholar: {'scholar': 50}; oracle: {'oracle': 49, 'persona': 1}; persona: {'persona': 47, 'oracle': 3}; mixed: {}

**unified8b_run2b failure categories:** judge: not consistently first person (17); judge: generic prose (17); judge: oracle quotes not relevant (8); judge: contradicts documented stance (7); judge: N/N scholar sentences unsupported (3); mode oracle != expected persona (3); N third-person self-reference(s) (3); N uncited sentences dropped (1); mode persona != expected oracle (1); judge: anachronism (1)
