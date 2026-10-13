# Replacement-loss v9 GMan V4e results

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4e.

## Outcome

- Execution status: INCONCLUSIVE_NO_EVALUATION.
- Reviewer status: NOT_SUBMITTED.
- Reproducibility status: INCONCLUSIVE.
- Packet SHA-256: 7b24806d2a6d3c97ca9579e10c1c0136a5415c08d940b69583cfa0f73886e914.
- The frozen V9 target remains FAIL; V4e did not evaluate or change it.
- GMan mission: https://autoresearch.sfcompute.com/missions/theseus-v9-linux-runtime-reproduction-20260928-v4e.

V4e transport preflight passed and held-out data stayed sealed. Its runtime
preflight stopped because the V9 runner enumerated vendored setuptools
packaging metadata at 26.0 instead of the active site-packages distribution
at 26.3. A read-only diagnostic on the same snapshot confirmed the other 50
active distribution versions matched the bound 51-package runtime.

## Receipts and custody

- Stop file SHA-256: 265f7dc9f966267a431d009f073f9ca8de3f8e36813bf5e966e0535d685cff4d.
- Stop receipt digest: 4c6ed337a6ccd555ac983afedc274c3aca9f33264165843b9024cceab3bfafb5.
- GMan stop artifact: art-5kkih.
- Transport file artifact: art-y5yez, SHA-256 3192dbaada484be1b02975ad7c1078f72e966066c2ae870551665aa2a0b34cab.
- Transport receipt digest: 5776d2639ddb835e79718f215cf01ae0f65fa52f8a2d2ddd88c322a9f3ca12eb.
- Both V4e nodes are stopped. The 15.9 GB source snapshot snap-t3r2t remains
  ready in private GMan custody.

## Spend and limits

V4e node usage was 4.03 minutes / $0.0324; its diagnostic node usage was
8.13 minutes / $0.0652. V4e-related incremental spend was $0.0976. Across
the Theseus V1, V3, V4, V4e, and V4e diagnostic nodes, settled spend was
$1.1422 of the $2.00 aggregate authorization.

The maximum claim is limited to a future same-owner local reproduction under
a frozen runtime. V4e established neither reproduction nor any scientific
outcome beyond its preflight failure.

