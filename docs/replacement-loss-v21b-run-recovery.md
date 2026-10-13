# Replacement-loss V21B: pre-data run recovery

State slices: `theseus-lab-sandbox-v1` and
`theseus-replacement-loss-v21b-run-recovery-v1`.

This continuation binds the frozen V21 development packet and its first
primary stop. The V21 runner attempted to read
`replacement_loss_v20b_runner.DEV_PRIMARY_NAME`, a constant that module does
not export. The terminal receipt is `INCONCLUSIVE`; no adaptation fixtures,
captures, candidates, or scores were produced in that V21 root. Preserve the
entire root and do not reuse it.

The recovery wrapper declares the exact existing V20B development-primary
custody path as an operator constant, redirects V21 development and
confirmation attempts to fresh V21B roots, and invokes the unchanged frozen
V21 runner. It does not change the V21 target, method, fixtures, thresholds,
runtime, controls, or claim ceiling. The wrapper, this recovery note, the
V21 packet, and the original terminal receipt are digest-bound before a new
fixture is generated.

If the recovery fails, retain its partial roots and receipts. Do not edit or
retry either frozen packet in place. Any source or scientific-method change
requires another named continuation and packet.
