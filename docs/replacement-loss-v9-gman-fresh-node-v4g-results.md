# Replacement-loss V9 GMan fresh-node V4g results

State slices: `theseus-lab-sandbox-v1` and
`theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4g`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

## Disposition

Status: `INCONCLUSIVE_NO_EVALUATION`.

V4g's frozen packet digest is
`70bc2450571e8477e45f0a9feff4414a5ea2899d4092c0e5d3efdaeba2a3bdc6`.
The checkout revision was `202f459d6d89df9a038d78d58e0af825d287fcd8`.
All five uploaded packet files matched their local SHA-256 values. The
custody importer passed and recorded that held-out predictions and outcomes
remained unopened. Runtime preflight stopped before writing runtime receipts
or starting the evaluator:

`V3 clone primary report differs from both frozen and known post-error bytes`

The 581-byte stop record has raw SHA-256
`a238e7eaf57ccb980c417fc65ecc798c43cf165d2b7db3dc95451ad0baa89feb` and
canonical stop digest
`5681fbc64ba0cd30e7f6b5667ed96b849a466748776cac2b580c9eecc53e04bc`.
GMan artifact: `art-2vc48`. The importer receipt has canonical digest
`0bf829e88ec2276143172ee2facd1c1d52e65c9bfef08db5fc54b5bf87089d69` and
GMan artifact `art-r54np`.

No candidate fitting, held-out prediction, evaluator run, or held-out outcome
opening occurred. V4g is terminal under its one-attempt rule; it was not
restarted or retuned. The node is stopped.

## Required report fields

- **Assurance:** `INCONCLUSIVE`; this is not `INTERNAL_PASS` or
  `REPRODUCED_LOCAL`.
- **Reviewer:** `NOT_SUBMITTED`.
- **Reproducibility:** `INCONCLUSIVE`; there was no completed comparison.
- **Custody:** the target archive, model assets, and data stayed in GMan.
  Only packet code and protocol files were uploaded from this checkout. The
  stop and transport receipts remain GMan artifacts; no prompts, traces,
  weights, or raw evaluation outputs were copied into this repository.
- **Cost:** GMan recorded `$0.0038` for the V4g node. Related V9 GMan node
  spend is `$1.1670` of the user-authorized `$2.00` ceiling, leaving `$0.8330`.
- **Known limitation:** the cloned primary report's current bytes do not match
  either digest allowed by V4g. This is a custody-integrity stop, not a
  scientific comparison.
- **Claim ceiling:** no V4g replication claim. The broader V9 result remains
  at most the previously recorded local result; this run adds no reproduction
  evidence. No claim is made about independent science, novelty priority,
  architecture-general validity, consciousness, alignment, human equivalence,
  or production authorization.
- **Next gate:** a new, separately frozen continuation must preserve the
  discrepant report bytes, verify the source archive and its unique expected
  report member, and restore that report from the verified archive before
  repeating preflight. Its target, scientific thresholds, fixtures, and
  evaluation order must remain bound to the existing V9 packet. The evaluator
  stays closed until the new preflight passes.
