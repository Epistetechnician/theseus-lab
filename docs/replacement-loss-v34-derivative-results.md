# V34 adaptation-only derivative diagnosis results

State slices: `theseus-lab-sandbox-v1`,
`theseus-v34-adaptation-jacobian-audit-v1`,
`global-experiment-assurance-lifecycle-v1`. Revision
`202f459d6d89df9a038d78d58e0af825d287fcd8` plus frozen packet.
As of 2026-10-03. [Protocol](replacement-loss-v34-derivative-audit.md),
[aggregate receipt](replacement-loss-v34-derivative-results.json).

## Outcome

`DIAGNOSTIC_SUPPORTS_ENDPOINT_OR_DERIVATIVE_HYPOTHESIS`.
Assurance/reproduction: `REPRODUCED_LOCAL`; reviewer `NOT_SUBMITTED`.
All three existing ReLU guides met the frozen diagnostic rule; all
decomposition controls passed. No optimizer updates and no challenge
evaluation occurred. V34's failed scientific outcome remains unchanged.

Primary cells: global arm, block-one stage, first activation site.

| Guide | Activation nRMSE | Prefix Jacobian nRMSE | Weak median effect error | Other median error | Secant/error Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| relu-34101 | 0.046 | 0.277 | 0.374 | 0.060 | 0.916 |
| relu-34102 | 0.046 | 0.270 | 0.333 | 0.055 | 0.938 |
| relu-34103 | 0.046 | 0.276 | 0.334 | 0.057 | 0.924 |

The weak-effect excess was .277–.315. Suffix-error RMS was exactly zero
in these shared-suffix cells. Coordinate-delta error RMS (.00712–.00761)
exceeded the base-shift term (.00338–.00424). Their vectors can correlate;
the RMS terms are not additive causal percentages.

Boundary/error rank correlations were .034–.062, while secant/error
correlations were .916–.938. This points to activation-difference precision
as a useful development hypothesis. Shared denominators and row dependence
limit causal interpretation of those associations. Donor activation nRMSE
was .0476–.0484, close to the base error; there was no large donor-only
collapse. The new donor endpoint arm is a control, not a diagnosed cure.

Guide local linearization differed from finite effects by .146–.192
normalized RMSE, so local derivatives alone need not capture every
boundary-crossing intervention. The diagnosis supports testing derivative
and secant supervision; it does not establish their heldout effectiveness.

## Validation and custody

Eight engineering tests passed per runtime, including analytic Jacobian versus
autograd checks for ReLU, GELU, SiLU-gated; prefix chain rule; zero subgradient;
trainable derivative losses; exact decomposition/shared suffix; changed suffix;
tie ranks; decision controls; invalid roles and terminal-root protection.

45 adaptation diagnostic cells and all array hashes, metrics, decisions, and
80-file manifests per run matched exactly. All 11 reconciliation checks passed.
Frozen archives contain 34 members plus the freeze receipt. The input manifest
binds 33 preparation/model files, and only adaptation/donor/mask fields were
loaded from preparation containers. Parent manifest verification may hash
other bytes; no challenge arrays or heldout targets were evaluated.

- Packet: `932443312451ba4e5cc67d7eac4f743e61ae8f1d97edb076f4633766a6ea9e84`.
- Freeze: `df2782b07c35a83c5585ecb9ee4b4900c4245e83b97722ff53256b03f14957cd`.
- Primary result: `b607a710d3f98c17a549e47874c4c68a24e638079081b9251747ddf8c5f3e4d9`.
- Replica result: `54572df4f1e639e4ff57af6afc8aa66dcdab1dce3d0106bfacd3124ef86f2e22`.
- Reconciliation: `a7f6fb94a672f253c4749aaed7edb7cc73e69fed5fe093302820fc2647f53cab`.
- Closeout validation: `f4aafce31e07979bb71bb6c97b28ef8aedcdecd1da46b5edb7824fd5f5edff1e`.
- Input manifest: `b198ddafb2be62d4255b42f2fdc165507b2411bef1f7613dac246efb0c18d597`.

| Runtime | Readout seconds | Peak RSS bytes | Final artifact bytes |
| --- | ---: | ---: | ---: |
| Primary | 2.24 | 240910336 | 130906458 |
| Replica | 2.15 | 240320512 | 130907059 |

Monetary cost $0; energy/amortization unmeasured. Both runs used unchanged
V34 isolated runtimes on the same host under the same owner. Raw row-level
vectors/Jacobians and frozen archives stay at
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-v34-derivative-audit`.
Only scalar/digest reports enter the repo. No independent authority or
external blind replication is claimed.

## Next scope and claim ceiling

The reproduced predeclared decision permits a fresh V35 development packet
for donor endpoints, coordinate-scaled secants, and prefix Jacobians, with
fresh globally disjoint inputs and unchanged all-stage causal gates.
No continuation is trained within this diagnostic or V34 identity.

Claim ceiling `ADAPTATION_ONLY_ENDPOINT_AND_DERIVATIVE_DIAGNOSIS`.
No claim of training causation, qualifying learned replacement, heldout
predictive success, pretrained transfer, all architectures, novelty,
metaphysical identity, consciousness, human-equivalent alignment,
independent science, or production authorization. Independent scientific
review, human calibration, and production remain separate future gates.

