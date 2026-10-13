# Replacement-loss v9 GMan V4f results

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4f.

## Outcome

- Execution status: INCONCLUSIVE_NO_EVALUATION.
- Assurance level: none.
- Reviewer status: NOT_SUBMITTED.
- Reproducibility status: INCONCLUSIVE.
- Packet SHA-256: ecc1bf4b55f17989685ca0c6cd9b4404b8cc942ee2602fabe64e26e3a27582fb.
- GMan mission: https://autoresearch.sfcompute.com/missions/theseus-v9-linux-runtime-reproduction-20260928-v4f.

The importer stopped on a parent snapshot-lineage check. The V4e freeze
correctly binds snap-jdsv3 as its original source snapshot; V4f incorrectly
compared it with snap-t3r2t, the later V4e workspace snapshot used to restore
the V4f node. The failure is preserved in GMan artifact art-myjn6, 488 bytes,
SHA-256
4a3f7536d96c8d4cd3106c81af7bd8ee5b7bfffbccc260e6cbce21329a0516e2.

Runtime preflight, candidate fitting, and evaluator execution did not start.
No held-out predictions or outcomes were opened. The V4f node is stopped.

## Spend and claim boundary

GMan reports 1.37 CPU-8 node-minutes and $0.0108 for V4f. Related settled
spend is $1.1530 against the $2.00 aggregate cap, leaving $0.8470 before
V4g.

V4f produced no scientific result. The next attempt uses a new frozen packet
with separate fields for the parent freeze's original source snapshot and
the later V4e workspace snapshot. The V9 target, fixtures, thresholds,
controls, and candidate procedure remain unchanged.

