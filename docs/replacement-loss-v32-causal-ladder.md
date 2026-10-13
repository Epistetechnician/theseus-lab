# Replacement-loss V32: trained nonlinear causal replacement ladder

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v32-causal-ladder-v1`,
`global-experiment-assurance-lifecycle-v1`. The bound historical audit uses
`theseus-all-runs-audit-v1`. Protocol: `theseus-replacement-loss-v32-causal-ladder`.

## Hypothesis and end state

A matched activation-interchange objective, evaluated through the remaining
network, can retain causal and behavioral responses while **both full nonlinear
blocks** are successively replaced by independently initialized modules. This
is a new synthetic trained-model study, motivated by V8's affine success,
V16–V21's activation/causality divergence, and V31's insensitive task.

End state: all nine predetermined cases execute or record their stop, all
predictions are bound before oracle outcomes, and the frozen procedure is
replayed in a distinct local environment. A failed scientific hypothesis can
still be REPRODUCED_LOCAL. There is no optional stopping, tuning, seed dropping,
new checkpoint selection, or rerunning a terminal custody root.

Claim ceiling: `LOCAL_SYNTHETIC_TRAINED_NONLINEAR_LADDER_CAUSAL_RETENTION_ONLY`.
No claim of a pretrained Transformer result, architecture-general validity,
novelty, metaphysical identity, consciousness, alignment, independent science,
production readiness, or publishing authority. Three activation families here
are **small feedforward guides**, not OPT/GPT-2/Llama checkpoints.

## Fixed task, models, and qualification

Input has eight real coordinates. Procedural outputs are
`[x0+x1*x2, x3-x4*x5, sin(x0+x3), x6*x7+x1*x4]`. Coordinate signs are task
decisions; continuous values test functional retention. The nonlinear
interaction task has no downloaded data, human payload, or license dependency.

Guide: affine 8→16 stem, two 16→64→16 nonlinear blocks, affine 16→4 head.
There is no residual bypass. Families: ReLU, GELU, and SiLU-gated (separate
up and gate matrices). Seeds: 3201, 3202, 3203, with family offsets 0/10000/20000.
Guide training: 4,096 independent uniform [-1,1] rows, 2,400 AdamW updates,
minibatch 256, LR .002, weight decay .0001, gradient norm clip 1.

Visible qualification: 1,024 disjoint uniform rows. Every guide must attain
coordinate sign accuracy ≥.95, normalized output RMSE ≤.15, and removal of
**each** block must lower sign accuracy by ≥.20. If any guide fails, stop as
NO_QUALIFIED_GUIDE before fitting candidates or drawing challenge entropy.
This prevents V31's random/zero-module task success from being called retention.
The nine fixed guide seeds are not selected by performance.

Replacement adaptation: 2,048 disjoint rows and 2,048 disjoint donor rows.
Train, visible, adaptation, donor, and heldout rows must be exactly disjoint;
the guide itself never trains on adaptation or heldout inputs. All arrays are
float32. Source nonlinear-block weights never initialize replacements.

## Four equal-budget methods and ladder

Every replacement is a new 16→96→16 GELU block. All four methods use identical
new-module initializations, adaptation rows, batch indices, AdamW LR .001,
weight decay .0001, batch 128, clipping 1, and 1,400 updates per conversion.
At each stage all replaced blocks are trainable, the remaining guide is frozen.
The second stage therefore learns on the already-converted prefix, not only on
original guide activations. Preserve snapshots after replacing block one,
block two, and final task training.

- `causal`: normalized guide output MSE + mean normalized activation MSE over
  converted block outputs + mean normalized intervention-effect MSE over those
  sites. Interventions replace one coordinate of a block output by the donor's
  value and run through the candidate's suffix. Targets come from the guide's
  matched interchange effects. Normalizers use target mean square, floor 1e-8.
- `pointwise`: same output and activation terms, without causal loss.
- `logit`: output-only distillation (continuous task outputs, not token logits).
- `cka`: mean row-normalized centered linear CKA dissimilarity over all
  converted block outputs. Staged schedule and similarity follow
  [NoT Algorithm 1 and Appendix B](https://arxiv.org/html/2512.04198v1).

Final task training: 600 additional updates with **all** candidate parameters
unfrozen, including copied stem and head. CKA uses task-output MSE only; other
methods keep their distillation losses and add task-output MSE. All use the
same adaptation task labels, initialization lineage, optimizer, and update
budget. This compares objectives, not total gradient-evaluation compute:
causal loss requires extra forwards, reported as such. NoT's progressive
conversion plus final task training is represented here; original paper data,
architectures, hyperparameters, and benchmark scores are not reproduced.
Interchange distillation overlaps with
[Causal Distillation](https://aclanthology.org/2022.naacl-main.318/); no priority
claim is made. There is no fitted SCM-to-hidden-variable abstraction claim.

## Heldout prediction and intervention semantics

Freeze every guide, candidate, fixture, and dependency identity before drawing
32-byte OS challenge entropy. Generate 256 fresh base/donor pairs per stratum:
ordinary, input composition, interchange composition, adversarial. The first
three use [-1,1]; adversarial uses [-1.4,1.4]. Every stratum tests two-coordinate
input substitutions plus neural output interchanges at both block boundaries.
Interchange masks replace respectively 1, 2, 4, and 8 of 16 coordinates; training
used only one. Neither heldout donor inputs nor combined masks enter training.

Interventions are explicitly **neural block-output** interventions in fixed
16-dimensional coordinates. Input substitutions have the procedural task's
ordinary counterfactual meaning. Internal swaps need not have an interpretable
task-level SCM meaning. No attention routing, permutation sweep, or parameter
renaming experiment is performed.

For each snapshot and control, bind ordinary outputs, input-intervention
effects, internal-intervention effects, block activations, and learned rescue
predictions to an aggregate digest before querying the guide on heldout rows.
The same process has both roles; this ordering is auditable but is not an
independent blind evaluator. The candidate predicts a restored output using
its own suffix. Only after prediction sealing, the evaluator fills an ablated
guide block-output slot with the candidate's corresponding activation and runs
the original suffix/head. Both restoration and prediction fidelity are scored.
Unchanged-guide restoration is an oracle positive control,
reported separately; it cannot count as learned rescue.

## Frozen gates and analysis

All nine causal-method cases, all three snapshots, and all four strata must
pass. Do not substitute averages for failed cells.

- Output normalized RMSE ≤.10; sign agreement with guide ≥.95; coordinate
  task accuracy drop relative to guide ≤.03.
- Input and active-block internal effects: guide effect L2 norm ≥1e-4 on at
  least 90% of rows; median relative L2 error ≤.20, 90th percentile ≤.50,
  median cosine ≥.90. Denominators floored at1e-6, not silently filtered.
- Normalized block activation RMSE ≤.15, learned rescue output RMSE ≤.10, and
  predicted-versus-observed rescue normalized RMSE ≤.10.
- Unchanged-guide positive control passes the same gates.
- Random fresh two-block, wrong-module block swap, and zero-module removal
  controls: input-effect median error exceeds causal block-two error by ≥.10
  in every stratum. Report all other control outcomes, including internal
  effect errors. Zero-effect prediction has error 1 on resolvable rows by
  definition and is an engineering metric control, not a trained arm.

Overall PASS requires every gate in every causal-method case. Comparative
superiority is **not** required for the primary retention claim. Report each
comparator's complete retention and paired case-level input-effect differences
after block two; no superiority or novelty conclusion follows without a
separately justified statistical/benchmark claim. Variance comes from guide
training, replacement optimization, and challenge contexts; all three seeds
are retained and methods share noise streams. No search or hyperparameter sweep.

## Runtime, custody, budget, and reproduction

Python 3.12.13; exact 11-distribution requirements lock; deterministic CPU
float32, one intra/inter-op Torch thread. Two newly created isolated environments
with distinct sys.prefix and site-packages; shared base interpreter permitted.
Per-runtime cap: 1,800 seconds, 4 GiB peak RSS, 1 GiB generated artifacts.
Training checks time/RSS every 50 updates; check artifact size between phases.
Monetary cap $0; local energy is NOT_MEASURED. No GMan/cloud/provider job is
needed or authorized by this packet. Extra causal forward-pass cost is counted
in wall/RSS but cannot be called an equal-FLOP comparison.

All generated data, weights, candidates, entropy, predictions, oracle outcomes,
and raw receipts remain under external owner custody in fresh V32 primary and
replica roots. Owner-only creation umask 077. Repository receives code,
protocol, frozen manifests, and aggregate results only. Archive every packet
member before generation. On any digest, dependency, disjointness, budget,
partial execution, or replay mismatch, retain receipts and report INCONCLUSIVE.

Replica reconstructs guides/candidates from the frozen procedure and seeds,
verifies preparation **before** opening the exact primary entropy, reconstructs
queries/predictions/outcomes, and compares canonical on-disk JSON plus array
content digests. Prefix paths, timing, and RSS are recorded but not equality
keys. Full procedure equality requires identical source, packages, platform,
thread count, fixtures, candidates, predictions, oracle arrays, and outcomes.
Source is reverified at completion. Same-owner separate-runtime equality is
REPRODUCED_LOCAL, not external acceptance. Reviewer: NOT_SUBMITTED. Local
retention can proceed under the user's authorization; independent science,
human alignment, consciousness research, and production have separate gates.
