# Replacement-loss V15b execution receipt

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v15b-capacity-optimization-diagnostic-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Disposition

The V15b diagnostic stopped before adaptation fixture generation, adaptation
capture, or candidate fitting. The runner loaded the bound OPT checkpoint and
generated its visible-only synthetic fixture, then raised `AttributeError`
because `_batches` is defined by `replacement_loss_v3_runner`, not
`replacement_loss_v3`. This is a startup-path failure, not evidence about
rank capacity or optimization.

- Frozen packet: `1fbf6d62ff00c984e3181991a0b9d61863d3dcbec4f62ddcec031dc59fafeabc`
- Freeze record: `30a612094ca36a99addd08e2b31f0808ef863b79b9d72864d09965044b32df1a`
- External custody root: `theseus-lab-sandbox-v1-replacement-loss-v15b-capacity-optimization-primary`
- Archive manifest SHA-256: `871cd673403487a75a1ec650d7473746d27a6e78f6efddfb557f3fe1ffd4a50a`
- Run-stop receipt SHA-256: `d5241768d2049b4f30fa10073d0e81fa21f39313c9008b2200bd1ef11cf3e6c7`
- Phase ledger SHA-256: `9413e16436333cbbf2e5c62167848e1b38417eb1ccc2af3a969d995eff3143bd`
- Visible fixture file SHA-256: `bf32c724f0cd3c3ea65a1ad485194bb0eb6e0d78b137c3276e8fc6cb0e3db907`
- Visible fixture example-set SHA-256: `b0ab8677b2418a32f49ef878382bfd38dd83f200da7eb44207c9e2b6cad2f014`
- Runtime: Python 3.12.13, Torch 2.6.0, Transformers 4.57.3, macOS ARM64
- Elapsed diagnostic phase: 1.822 seconds; peak RSS: 461,029,376 bytes

The generated 32-row visible synthetic fixture stays in external owner
custody. No adaptation fixtures, adaptation captures, candidate weights,
predictions, or training metrics were generated. The V15b custody root and
receipt are preserved and must not be reused.

## Assurance and claim boundary

- Assurance: `INCONCLUSIVE`
- Reviewer status: `NOT_SUBMITTED`
- Reproducibility: not assessed; the diagnostic did not reach adaptation
- Data and artifact custody: external owner custody on `/Volumes/PrimaryED`
- Scientific claim: no result about replacement capacity, optimization,
  behavior, retention, or Theseus continuity
- Claim ceiling: visible-fixture generation and runner startup only
- Next gate: a fresh continuation packet that binds every V15 utility call to
  its actual owning module before generating new fixtures
