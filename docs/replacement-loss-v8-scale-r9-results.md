# Replacement-loss-v8-scale-r9 results: news-domain replication

State slice `theseus-lab-sandbox-v1`. Protocol `theseus-replacement-loss-v8-scale-r9-news-domain`, packet `c41dbecc347754f4e3f96fd29b526bd5f6b1c65582d6f42939e917c7df2d5e6c` (105 files; second freeze; identical in both arms). Reports: `docs/replacement-loss-v8-scale-r9-{NN256,NN512}-report.json`. Preregistration: `docs/replacement-loss-v8-scale-r9.md`.

## Classification (preregistered rule): `REPLICATES`

Per-case result (all nine cases pass the frozen per-case gates, plus the three cluster-bootstrap gates) is a pass at both arms, on CNN/DailyMail news instead of wikitext-2.

| arm | adaptation / challenge | per-case + bootstrap | frozen arm `outcome_status` | 1e-4 equivalence diagnostic | peak RSS |
|---|---|---|---|---|---:|
| NN256 | 256 news / news | **PASS, 9/9** | `FAIL` | fails in 3 cases (SmolLM2 seeds 0,1,2) | 19.8 GiB |
| NN512 | 512 news / news | **PASS, 9/9** | `FAIL` | fails in 2 cases (SmolLM2 seeds 0,1) | 25.0 GiB |

The arm-level `FAIL` comes only from the frozen 1e-4 diagnostic and is reported unaltered; it is not a PASS of either arm. No resource blockers.

Diagnostic misses (module-effect max abs error; logit error is 3-5e-5 everywhere, within bound): NN256 SmolLM2 2.4e-4, 3.1e-4, 3.7e-4; NN512 SmolLM2 2.4e-4, 6.1e-4. opt-2.7b and gpt2-xl are within 1e-4 in all cases of both arms (gpt2-xl module errors up to 7.6e-5 in the discarded first-freeze run, close to the bound).

Mean (over 3 seeds) pooled median cosine of predicted vs observed logit effect, frozen case gate 0.90, all cases pass:

| model | r9 NN256 | r9 NN512 | r6 NN256 (wikitext) | r6 NN512 (wikitext) |
|---|---:|---:|---:|---:|
| opt-2.7b | 0.9997 | 0.9998 | 1.000 | 1.000 |
| gpt2-xl | 0.9996 | 0.9996 | 1.000 | 1.000 |
| SmolLM2-1.7B | 0.9806 | 0.9795 | 1.000 | 1.000 |

Per-stratum median cosine across cases: min 0.960 (NN256 compositional), 0.969 (NN512 compositional); medians 0.998-1.000.

## Predictions

- P1 NN512 per-case pass: **held**. P2 NN256 per-case pass: **held**.
- P3 (pooled median cosine >= 0.99 in every model/arm): **missed** for SmolLM2 (0.98 in both arms; individual seeds 0.968-0.984). Gates still pass (0.90). On wikitext r6 SmolLM2 was 1.000, so there is a domain effect on prediction quality in this model, not on pass/fail. Cause not investigated.
- P4 (diagnostic fails only in SmolLM2): **held**; r7's float32 explanation was not retested here.
- P5 (peak RSS < 30 GiB): **held** (25.0 GiB max).

## Readings

The 256+/512 natural-text replacement result is not specific to wikitext-2: it also passes on news text for all three architectures and all three seeds at 256 and 512 prompts. No threshold shift between arms (both pass). The SmolLM2 cosine drop is the only visible sign of domain dependence, and the frozen 1e-4 diagnostic misses SmolLM2 in the same way as r6 (exceeding the bound but not the case gates). This is two domains, not natural text in general.

## Limits and disclosures

- **Pre-run corrections / reruns (all before any reported result):** (1) first freeze (`bdb963e6…`): NN256 ran (9/9 per-case, same pattern; discarded, kept in the notes folder as `r9-first-freeze-NN256-report-DISCARDED.json`); NN512 crashed at the fixture-disjointness preflight, before models or challenges, because syndicated articles share identical 45-word ledes. Fixed by also de-duplicating on the emitted 45-word prompt (no-op on wikitext), refrozen, and both arms rerun from scratch; this changes the filter relative to "mirror the wikitext filter" by one condition, stated in the prereg doc. (2) The r5-n512 node was reclaimed (returned to queued, one run exited with no output, no result) during the first NN512 attempt; NN256 and the NN512 rerun both ran sequentially on r5-n256 under the second freeze instead of in parallel on two nodes. Challenge entropy is fresh per run.
- Both reported arms ran on one node, same custody models copied from r5-N256, packet verified identical (105 files).
- Corpus: validation split of one dataset, English news with wire/byline boilerplate left in; articles truncated to their first 45 words; first-45-words prompts over-represent ledes. Same three checkpoints, 3 seeds, CPU fp32, one projection each, synthetic intervention directions, same owner, no independent review. Hypotheses until independently validated.
- r6's own test `test_natural_adaptation_reuses_r5_prompts_exactly` needs `THESEUS_WIKITEXT_PARQUET` and errors without it; unrelated to r9.
- Spend (from `gman usage`): r5-n256 $3.14 + r5-n512 $0.22 = $3.36 for r9, within the $15 cap. Billing balance moved $506.77 -> $499.91, which also includes other sessions' nodes running concurrently.

## Addendum: CI note (added when landing)
`tests/test_replacement_loss_v8s9.py::test_pinned_chain_covers_r5_to_r7_and_band_b_and_matches_the_freeze` asserts that no r8 files are tracked (true on the branch the packet was frozen from, where r8 had no files on main). r8 was merged to main before r9, so that scope guard fails on the merged tree. The test and the frozen packet were not changed. `scripts/run_public_tests.sh` reports the module as SKIPPED-OWNER-ENV; run it on the r9 branch snapshot (or with `STRICT=1` there) to exercise it. The r9 chain therefore does not include r8's files.
