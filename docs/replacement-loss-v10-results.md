# Replacement-loss v10 results

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v10-opt-causal-fit-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol: `theseus-replacement-loss-v10-opt-causal-fit`.

## Outcome

Scientific outcome: **`INCONCLUSIVE`**. The frozen checkpoint screen found no
eligible fresh OPT/ReLU checkpoint, so the run stopped before checkpoint
selection, fixture generation, candidate fitting, challenge prediction, or
scoring. This is the stop condition specified by the frozen protocol.

The screen reviewed 13 candidate records and found zero eligible candidates.
Its Hugging Face metadata query returned a bounded first 1,000 OPT-tagged
entries; the search was not exhaustive. For example, `facebook/opt-125m` met
the architecture shape but its identity is excluded by the frozen packet;
the reviewed alternatives either had GELU, lacked declared license or base
checkpoint provenance, or described fine-tuned or modified derivatives.
No checkpoint was selected.

## Frozen packet and screen evidence

- Source revision (`HEAD`):
  `202f459d6d89df9a038d78d58e0af825d287fcd8`.
- Frozen packet SHA-256:
  `02c20d0953c93d088887c1a0494131504e3f17f7018b9d5557ebc884b53d6475`.
- External packet archive SHA-256:
  `0bd878b21cf8441ef587b36444102fa80c8e116790778d2f3ad3da6fb60118cb`.
- External screen report file SHA-256:
  `5704b19b3dbc544a29fad7a4808ebfc3812a7b59a4e86a13621c9b1d3619775c`.
- The screen report's `report_sha256` field is its canonical digest with that
  field omitted: `6a29ac55f1c9323a49f04e15f55a18b2c0b6e68a605266b76bba2e3ce26d26d2`.
- The frozen source packet and all 10 archived packet files passed digest
  verification after screening.

Before freeze, the focused V10 test suite passed 9/9 cases in both pinned
local Python 3.12.13 runtimes (PyTorch 2.6.0, Transformers 4.57.3); the lock
matched both environments. These tests used constructed models and qualify
the runner contract only. Scientific-run resource use is
`NOT_APPLICABLE_NO_SCIENTIFIC_RUN`; energy is `NOT_MEASURED` and monetary cost
is `NOT_APPLICABLE`.

The screen retrieved public model-card, configuration, and weight-file
metadata. No model weight bytes were downloaded. No prompts, synthetic
fixtures, captures, candidates, held-out predictions, or outcomes were
generated. No experiment phase completed. Paid and remote compute were not
used.

## Assurance and custody

- Assurance: `INTERNAL_PASS` for the frozen packet checks, hermetic runner
  qualification, and bounded metadata screen. It is not a scientific pass.
- Reviewer status: `NOT_SUBMITTED`.
- Reproducibility status: `NOT_APPLICABLE_NO_SCIENTIFIC_RUN`.
- Data and artifact custody: model-card/configuration metadata and the screen
  report remain in owner-controlled external custody. No model or experiment
  payload was placed in this repository.
- Known limitation: the bounded catalog search cannot establish that no
  eligible checkpoint exists elsewhere in the catalog.

Claim ceiling:
`LOCAL_CAUSAL_LOSS_IMPROVEMENT_FOR_TWO_VISIBLE_SELECTED_OPT_PROJECTIONS_ON_ONE_CHECKPOINT_AND_THREE_SYNTHETIC_SEEDS`.
No causal-loss improvement claim was measured because no model run occurred.
This packet does not support architecture-general validity, general
mechanistic understanding, novelty priority, whole-model identity,
consciousness, alignment, human equivalence, independent science, or
production readiness.

## Next gate

Verify and select a fresh, public, ungated base OPT/ReLU checkpoint with a
declared license and provenance that passes the frozen freshness rules. Once
that gate is satisfied, the frozen local procedure can continue. If the
checkpoint class or eligibility rules need to change, freeze a new protocol
before selecting a target.
