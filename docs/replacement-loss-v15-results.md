# Replacement-loss V15 execution receipt

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v15-capacity-optimization-diagnostic-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Disposition

The frozen V15 diagnostic did not execute. Its single primary attempt stopped
before checkpoint loading, fixture generation, activation capture, or
candidate training. The terminal status is `INCONCLUSIVE`; this is a runner
startup failure, not a scientific result about replacement capacity or
optimization.

- Frozen packet: `d72a6fd520a185fbfeacf63e46c55ec43ef273017fd2bc8ab103b8165efbc042`
- Freeze record: `faec9cfc11e50608f6e5d43cf15860d5580f0427688db3443e0151a35f703d1d`
- External custody root: `theseus-lab-sandbox-v1-replacement-loss-v15-capacity-optimization-primary`
- Archive manifest SHA-256: `af56f0ea273ec81f589c6c9852b52286614d4192c934194f1bc8b4b59685f3ee`
- Run-stop receipt SHA-256: `9594780fc937d6ccd92e87dc00f1e85f65078ec1c5699d67f973d69b401ef9e4`
- Phase ledger SHA-256: `c94e3ad2c2952b669f830b4caaf473ef122342e79f1143eb8e57f13a358a99ae`
- Runtime: Python 3.12.13, Torch 2.6.0, Transformers 4.57.3, macOS ARM64
- Elapsed diagnostic phase: 0.019 seconds; peak RSS: 280,264,704 bytes
- Error: `AttributeError` because V15 looked up `Checkpoint` in
  `replacement_loss_v3`, while the checkpoint type is provided by
  `replacement_loss_v3_runner`. Source inspection found the adjacent local
  model-loader reference pointed at the same utility module and would fail
  after the first lookup was corrected; V15b binds both functions from their
  owning runner module.

The custody root contains the archived frozen packet, target-selection
binding, phase ledger, and stop receipt. It contains no generated fixture,
capture, prediction, metric, or candidate artifacts. The root and receipt are
preserved and must not be reused.

## Assurance and claim boundary

- Assurance: `INCONCLUSIVE`
- Reviewer status: `NOT_SUBMITTED`
- Reproducibility: not assessed; there was no experiment execution to replay
- Data and artifact custody: external owner custody on `/Volumes/PrimaryED`
- Scientific claim: none about model behavior, capacity, optimization,
  replacement retention, or Theseus continuity
- Claim ceiling: startup-path validation only
- Next gate: a fresh continuation packet with corrected model-loader binding,
  frozen before any new fixture generation or fitting

The correction is carried in the separately named V15b continuation packet;
the frozen V15 runner, packet, and failed receipt remain unchanged.
