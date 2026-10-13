# Replacement-loss v11: failure-informed causal-fit remediation on a reused target

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v11-known-opt-causal-fit-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v11-known-opt-causal-fit-remediation`.

## Purpose and claim ceiling

V10 remains frozen and `INCONCLUSIVE` because its post-freeze target screen
verified no fresh eligible OPT/ReLU checkpoint. V11 addresses the more
focused question enabled by existing evidence: does adaptation-only causal
loss improve held-out causal-effect prediction over activation ridge on new
synthetic fixtures for the exact OPT/ReLU checkpoint that produced V9's
known causal-effect misses?

This is an intentional reuse of
`grushaprasad/opt125m-rootp-surprisal-curriculum@8af2a2926bc772599ccb3b4e7ad7b40b35259a63`.
Its identity, revision, license record, checkpoint tree digest, and weight
file digest are bound before V11 fixtures are generated. V11 makes no
fresh-checkpoint generalization claim.

The maximum positive claim is
`LOCAL_CAUSAL_FIT_REMEDIATION_ON_REUSED_V9_OPT_CHECKPOINT_WITH_NEW_SYNTHETIC_FIXTURES`.
This experiment cannot establish novelty priority, architecture-general
validity, whole-model identity, general mechanistic understanding,
consciousness, alignment, human equivalence, independent science, or
production readiness.

## Hypothesis and outcomes

The primary hypothesis is that `causal_fit` will improve mean held-out
next-logit intervention-effect cosine over `activation_ridge` by more than
0.05 at the frozen 95% lower confidence bound, while meeting the V10 causal,
behavior, activation, activation-rescue, and no-op retention gates.

The primary metric, bootstrap unit, 10,000 draws, fixed bootstrap seed, and
0.05 threshold are inherited unchanged from the V10 engine. Three paired
fixture seeds are the independent clusters; layers and prompts are not
treated as independent samples. Main variance sources are fixture seed,
candidate initialization and batch order, and numeric runtime differences.

V9 recorded next-logit causal-effect misses at ordinary strata for layers 0
and 7 and at the adversarial stratum for layer 0, across all three V9 seeds.
V11 fixes layers 0 and 7 before freeze. It retains the remaining strata as
within-target coverage. A predeclared secondary summary reports how many of
those nine previously missed seed/layer/stratum cells meet the causal-effect
gate under each method. That summary is descriptive and does not add another
primary pass condition.

Controls are the V10 engine's randomly initialized replacement,
wrong-layer replacement, and zero-intervention check. The comparator is the
same-seed, same-layer `activation_ridge` candidate. Adaptation and held-out
challenge prompts use new V11 namespaces and three new fixture seeds. The
challenge entropy is generated only after candidate states are frozen.

## Frozen procedure

1. Freeze this protocol, the fixture generator, the V11 adapter, the imported
   V10 engine and V3 support files, tests, and dependency lock. Bind the V9
   packet and checkpoint-manifest digests and the exact reused model identity
   in the freeze record.
2. Verify the already-custodied V9 checkpoint tree, license metadata, config,
   and raw weight-file digest. Do not download or copy its model weights.
   Persist only the V11 target-binding receipt in the new external custody
   root.
3. Run three seeds. Use 32 visible prompts for model inspection and 96
   adaptation prompts per seed (32 per stratum). Fit the same two methods as
   V10 independently at fixed layers 0 and 7.
4. Freeze both methods' candidate states. Only then generate 60 held-out
   challenge prompts per seed (20 per stratum) from fresh 256-bit entropy.
   Seal candidate/control predictions before opening teacher outcomes.
5. Score the primary comparison, all retention gates, activation rescue,
   controls, and the predeclared V9 failure-cell summary. Preserve the exact
   result, including `FAIL` or `INCONCLUSIVE`.
6. Re-run the frozen packet in the separate pinned local runtime using the
   exact same checkpoint and challenge entropy. Report `REPRODUCED_LOCAL`
   only if candidate, challenge, prediction, outcome, dependency, and source
   digests match.

No thresholds, seeds, target layers, prompts, entropy, or outcome gates may
change after freeze. A stop or failure is terminal for that custody root.

## Budget, custody, and gates

Execution is local CPU only, one thread, at most four hours total, at most
16 GiB resident memory, at most 200 million parameters, and at most 800 MB of
checkpoint weights. The checkpoint remains in existing owner-controlled V9
external custody; V11 stores only its manifest binding. Prompts, fixtures,
entropy, captures, candidates, predictions, interventions, outcomes, and
row-level data remain outside the repository. No paid compute, cloud worker,
GMan node, provider call, or public write is in scope. Monetary cost is
`NOT_APPLICABLE`; energy is `NOT_MEASURED`.

`INTERNAL_PASS` describes code and packet qualification. `REPRODUCED_LOCAL`
requires exact reproduction in the separate local environment and remains
same-owner evidence. Reviewer status is `NOT_SUBMITTED`; no independent or
production authority is implied.
