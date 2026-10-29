# Replacement-loss-v8-scale-r7 results: float64 check of the r6 equivalence-diagnostic misses

State slice `theseus-lab-sandbox-v1`. Protocol `theseus-replacement-loss-v8-scale-r7-equivalence-float64-diagnostic`, final freeze packet `6fbcbbde939a748d44f28d3fe5cb8259b4275fa126ec7605983e167255321ce7` (58 files). Reports: `docs/replacement-loss-v8-scale-r7-{NN256,NN512}-report.json`.

**The r6 NN256 and NN512 arm outcomes remain `FAIL`.** r7 is diagnostic only; no gate, constant or r6 file was changed or re-scored.

## Outcome (preregistered rules): `PARTIAL` in both arms

Measured on the original r6 custody (same fitted modules, same challenge prompts and interventions), SmolLM2-1.7B, layer 7, 3 seeds per arm. Max-abs error between treatment and mapping-aligned permutation predictions:

| arm / seed | fp32 module | fp64 module | fp32 logit | fp64 logit | max abs boundary output |
|---|---:|---:|---:|---:|---:|
| NN256 / 0 | 1.22e-4 | 2.3e-13 | 4.8e-5 | 5.2e-6 | 1614 |
| NN256 / 1 | 2.29e-5 | 9.9e-14 | 4.4e-5 | 5.9e-6 | 30 |
| NN256 / 2 | 6.10e-4 | 9.1e-13 | 4.0e-5 | 6.6e-6 | 1956 |
| NN512 / 0 | 8.54e-4 | 4.5e-12 | 4.3e-5 | 5.9e-6 | 1592 |
| NN512 / 1 | 1.53e-4 | 4.3e-12 | 3.6e-5 | 3.7e-6 | 1684 |
| NN512 / 2 | 1.53e-5 | 9.5e-14 | 4.5e-5 | 4.9e-6 | 28 |

- **Float32 reproduces r6 exactly:** the four over-threshold module errors are the r6 values (NN256 1.2e-4, 6.1e-4; NN512 8.5e-4, 1.5e-4).
- **P1 holds (6/6):** every float64 error, module and logit, is <= 1e-4 (module about 1e-13..4e-12).
- **P2 fails, so the label is PARTIAL:** P2 required every error to shrink >=100x where float32 exceeded 1e-4. The module error shrinks by 8-9 orders of magnitude, but the *logit-effect* error only shrinks about 7-12x, to a floor of 3.7e-6..6.6e-6 in float64. The prereg rule takes the max of module and logit errors, so P2 fails as written. I did not change the rule. The logit floor is **unexplained**: it is far below 1e-4, and no r6 miss was a logit error (float32 logit errors were 3.6e-5..4.8e-5, all within the bound), but a pure summation-order effect would be expected to vanish in float64 too. Candidate causes (not tested): a float32 step downstream of the projection output (e.g. a cast in the logit path), or deltas/inputs entering in float32.
- **P3 `SCALE_CONSISTENT` (both arms):** relative float32 module error is 7.6e-8..7.6e-7, <= 1e-5. This is only a ceiling check. The preregistered Spearman correlation between float32 error and max boundary activation is 1.0 (NN256) and 0.5 (NN512), n=3 per arm, not a significance claim. Note seed 2 of NN256 and seeds with max boundary outputs near 30 show errors 2e-5..1.5e-5 against 1.6e3-scale outputs giving 1e-4..9e-4, i.e. error does track activation magnitude, but one outlier-token-scale activation (max output ~1.6e3 vs a median output norm ~41) drives it.

## Reading

The 1e-4 absolute equivalence misses in r6 are float32 rounding in the module-effect comparison: same weights, same inputs, float64 gives ~1e-13. They are not evidence of a non-equivalent permutation mapping. This does **not** alter the frozen r6 outcome (the gate is absolute, in float32, as preregistered), and it is not a PASS of any arm. The unexplained 5e-6 logit-effect floor is a separate open item.

## Limits and disclosures

- Pre-run amendments (all before any measurement; in Git history): the first freeze (`8a5c4c8`) assumed regenerated inputs; refrozen (`e83b11e`) to a read-only `diagnose` mode on the intact r6 custody; a first launch then crashed at model load on the 500M-parameter ceiling, producing no measurements, and was refrozen (`4211a93`) with the pinned r6 parameter-ceiling profile. The final freeze is the one reported.
- SmolLM2 only (opt-2.7b and gpt2-xl had no misses), one projection, 6 cases, same owner, CPU, no independent review. The claim that the 1e-4 bound was calibrated on template activations was not tested (no template-scale comparison).
- Operational: overlapping background watchers once stopped the NN512 node early; no measurement was affected, the run was repeated cleanly.
- Spend: gman did not report realized cost. Rate was $0.012/min per cpu-8 node while running; two nodes, roughly 1 hour of running each plus restore/idle time, so an estimate of about $2-3, well under the $10 cap. Not measured.
