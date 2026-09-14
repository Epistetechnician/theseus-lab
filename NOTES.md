# Prototype notes

Question: Can a system preserve its observed behavior while its explicitly
tracked computational mechanism is replaced or lost?

Current result: the prototype separates two effects. A structural split keeps
visible and held-out behavior at `1.0` while mechanism continuity falls to
`0.4167`, producing a structure-only Theseus counterexample. A replacement
matches visible probes but fails held-out probes, demonstrating that visible
behavioral equivalence can be a probe-coverage artifact. This is a deterministic
state-machine observation, not evidence about a trained model.

The causal intervention score and component-set overlap are diagnostic signals;
they do not establish mechanistic understanding. This slice now routes the
synthetic checkpoint through `LocalCheckpointAdapter`, binds every state to a
full checkpoint-summary digest, and retains held-out probes, interventions, and
the explicit lineage ledger.

Real local checkpoint result: the pinned GPT-2 bundle loaded successfully from
external custody with all seven artifact digests matching, 160 tensors present,
and checkpoint digest
`bfca28eb087ca25477e3e9b05f1647cb47bbc2f90977a6b54101fabd0c9ca070`. A
layer-0 MLP permutation changed 100% of the selected parameter positions while
preserving top-1 behavior on all visible and held-out prompts; mean relative
logit error was `3.3e-7` visible and `1.2e-7` held out. Ablating the same channel
label after permutation produced cosine similarity `-0.98309803` against the
baseline intervention, while ablating the inverse-mapped channel produced
`0.99999964` cosine similarity.

Insight: channel labels and raw parameter positions are not stable evidence of
causal identity. A function-preserving reparameterization can make a same-label
intervention look unrelated even when the mapped intervention is nearly exact.
This supports a representation-relative version of the Theseus thesis, not a
claim of mechanistic understanding, general model identity, or production
readiness.

Independent reference and symmetry sweep result: PyTorch/Transformers matched
the NumPy implementation on tokenization, top-1 outputs, and mean relative
logit error `4.9e-7` with 95% bootstrap interval `[4.7e-7, 5.1e-7]` across 72
fixed prompts. Forty-eight MLP-channel and attention-head permutations across
all 12 layers passed. Every case preserved visible and held-out top-1
behavior, changed the selected parameter positions, recovered the mapping by
blind component signatures, and restored intervention effects under the
inverse mapping without relaxing the original cosine or maximum-absolute-error
thresholds.

The earlier float32-only run exposed accumulated roundoff in one late-layer
aligned intervention. The current runner keeps the broad behavior and atlas
path in float32 but recomputes the narrow intervention comparison in float64;
the original cosine and maximum-absolute-error thresholds remain unchanged.

Non-symmetry result on the 72-prompt set: 13 of 14 controls produced measurable
logit divergence. The remaining single-channel ablation is a bounded
probe-resolution negative at one late layer; it is not proof that the channel
is globally irrelevant. All 14 controls retained canonical digest
differences. This separates parameter-set change from observed sensitivity and
keeps ablation, pruning, and replacement outside the identity class.

The report claim ceiling for the original single-checkpoint run was
`LOCAL_GPT2_TRANSFORMATION_OBSERVATION_ONLY`. No general mechanistic,
scientific, alignment, or production claim follows from this prototype.

Formalized identity criterion result: all 48 symmetry cases (two shifts for
each MLP-channel and attention-head family at every layer) passed the stronger
gate: raw digest difference, blind canonical digest equality, blind mapping
recovery, visible plus held-out behavior match, and aligned intervention match.
The canonicalization is a deterministic sort of exact composite component
signatures; it does not consume the recorded permutation. The 14 ablation,
pruning, and replacement controls retain canonical digest differences and are
never accepted as identity-equivalent.

The expanded causal atlas contains 72 entries: three MLP channels and three
attention heads at each of all 12 layers. Every entry reports mean relative
logit effect and a deterministic 95% bootstrap interval over the 72 fixed
probes. Top-1 changes were sparse and sometimes absent despite a nonzero logit
effect, so top-1 alone is not treated as causal evidence.
The combined probe-set digest is
`ef4e0b6ff2c8358f1e9350b06aeaf15c11d87b6bddc83e79cd6d14901f1035bf`.

The local Jacobian-lens proxy fits on the eight visible prompts and evaluates
on the 64 held-out prompts in post-GELU MLP activation space. Its target tokens
are `(" the", " is", " of", " and")`, IDs `(262, 318, 286, 290)`, with target digest
`8c055c3ea9637aa48eeab068e2411adeeccf9180a04eb25f7e8c135583346ae6`. The
fit-prompt digest is
`f159840c3248df49b7591590a0e1979b04fc0c93e054062fafe7dc4f3f6cca12`, and the
64-probe evaluation digest is
`82fb130a5e31ccddf61eb452399a4a342d39649005a949e61b9ff314a3542f4c`.
permutation stability check passed at all three layers. Raw-label coordinate
relative errors were `0.56856283`, `4.99736691`, and `2.74498723` at layers 0,
5, and 11, while inverse-aligned errors were `0.0` at all three. Pruning 128
channels diverged in the proxy at every selected layer, with mean relative
errors `0.02940231`, `0.89672909`, and `0.97816896`.

The resulting insight is narrower and falsifiable: within this checkpoint and
probe contract, channel/head labels are not identity primitives; identity is
better modeled as an equivalence class under validated function-preserving
symmetries plus aligned causal interventions. Blind canonicalization and
internal-coordinate Jacobian measurements sharpen the result: raw internal
coordinates change substantially under relabeling, while inverse-aligned
coordinates recover the baseline and pruning diverges. The single-checkpoint
claim ceiling remains `LOCAL_GPT2_TRANSFORMATION_OBSERVATION_ONLY` plus
`LOCAL_JACOBIAN_PROXY_OBSERVATION_ONLY`; no general mechanistic, scientific,
alignment, or production claim follows.

Cross-checkpoint replication result: the same local protocol passed on both
the 12-layer GPT-2 checkpoint (revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, checkpoint digest
`4d4f6fe7cdf15eed927137a1fe8fd6af5dda4ae791c4730c850c5d1d76ad70bd`) and the
6-layer distilgpt2 checkpoint (revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, checkpoint digest
`b10648d21b2ff273da0f9f41c228ecbbded22773cc9664df554e5c13d050e5ab`). Both
used the same probe-set digest
`ef4e0b6ff2c8358f1e9350b06aeaf15c11d87b6bddc83e79cd6d14901f1035bf`.

GPT-2 passed 48/48 symmetry cases, recovered 48/48 blind mappings, retained
14/14 canonical control differences, and produced 72 atlas entries.
distilgpt2 passed 24/24 symmetry cases, recovered 24/24 blind mappings,
retained 8/8 canonical control differences, and produced 36 atlas entries.
Both passed the internal Jacobian proxy with raw-coordinate change,
inverse-aligned stability, and pruning divergence. The combined command
reported `CROSS_CHECKPOINT_REPLICATION_PASS`.

This is evidence that the observed equivalence-class behavior is not confined
to one depth of the GPT-2 family. It remains a two-checkpoint local result,
not architecture-general evidence, mechanistic understanding, alignment
validation, or production readiness. The family-level claim ceiling is
`LOCAL_GPT2_FAMILY_REPLICATION_OBSERVATION_ONLY`.

Out-of-family OPT-125M result: the pinned `facebook/opt-125m` bundle at
revision `27dcfa74d334bc871f3234de431e71c6eeba5dd6` loaded from external
owner-only custody. Its seven artifact digests and 197 state-dictionary
tensors were verified; the resulting checkpoint digest is
`6405b806c3c29e3f2a3700677b8e1d6d347b608abf444505bef6577a369e1679`.

The independent NumPy OPT decoder matched local PyTorch/Transformers on all
72 fixed probes with top-1 agreement `1.0`, mean relative L2 `5.5e-7`, and
95% bootstrap interval `[5.3e-7, 5.6e-7]`. The OPT run passed 48/48 symmetry
cases, recovered 48/48 blind mappings, retained 14/14 canonical control
differences, changed behavior in 14/14 controls, and produced 72 causal-atlas
entries. The local Jacobian proxy passed on layers 0, 5, and 11: raw-coordinate
relative errors were `2.48679844`, `2.10433353`, and `1.61749519`, while
inverse-aligned errors were `0.0` at all three layers. Pruning diverged at all
three layers.

OPT uses ReLU MLPs, so the inherited fixed channel 17 was degenerate on the
reference prompt. The corrected protocol selects one channel per layer using
only the eight visible probes, choosing the highest positive activation count
with a deterministic lowest-index tie break; it evaluates aligned interventions
on 16 fixed held-out probes. The selection and evaluation digests are
`f159840c3248df49b7591590a0e1979b04fc0c93e054062fafe7dc4f3f6cca12` and
`988cbed4e35a0b9397a880df683d7fe66996baf71c118f3d4c73b4b8f4d83f2d`.

The fresh three-checkpoint replay passed with the shared probe digest
`ef4e0b6ff2c8358f1e9350b06afae15c11d87b6bddc83e79cd6d14901f1035bf` and
reported `CROSS_ARCHITECTURE_REPLICATION_PASS`. This expands the observed
equivalence-class behavior beyond the GPT-2 family to the OPT architecture
under this exact local protocol. It remains local replication evidence, not
architecture-general proof, mechanistic understanding, alignment validation,
or production readiness. The cross-architecture claim ceiling is
`LOCAL_CROSS_ARCHITECTURE_REPLICATION_OBSERVATION_ONLY`.

Topology-diverse SmolLM-135M result: the pinned `HuggingFaceTB/SmolLM-135M`
bundle at revision `1d461723eec654e65efdc40cf49301c89c0c92f4` loaded from the
external owner-only custody root. The manifest verified eight bundle artifacts
and 272 state tensors; tied output embeddings were reconstructed from the
verified input embedding. The resulting checkpoint digest is
`8c0b435ce577250d0f9b2fb8334342ed1a65b2c8110eff3cbebc59b28a5097ad`.

The independent NumPy Llama-family decoder matched local PyTorch/Transformers
on all 72 fixed probes with top-1 agreement `1.0`, mean relative L2
`2.01e-6`, and 95% bootstrap interval `[1.74e-6, 2.32e-6]`. The architecture
has 30 layers, RMSNorm, rotary position encoding, a SiLU gated MLP, nine query
heads, and three key/value groups. The symmetry relation therefore includes
gated-MLP channel permutations and complete GQA-group permutations, with each
KV group moved together with its three associated query heads.

The SmolLM run passed 120/120 symmetry cases and recovered 120/120 blind
mappings. All 32 non-symmetry controls retained canonical digest differences
and changed behavior on the 72-probe suite. The causal atlas contains 90
gated-MLP channel entries and 90 GQA-group entries, each with deterministic
95% bootstrap intervals. The local Jacobian-lens proxy passed on layers 0, 14,
and 29 with raw-coordinate change, inverse-aligned stability, and pruning
divergence. Its claim ceiling is
`LOCAL_SMOLLM_TRANSFORMATION_OBSERVATION_ONLY`.

This is topology-diverse local evidence that the validated equivalence-class
criterion survives a Llama-family implementation with GQA, RMSNorm, rotary
position encoding, and gated MLPs. It is not architecture-general proof,
mechanistic understanding, alignment validation, or production readiness.

Four-checkpoint topology replay: the combined local runner passed
`CROSS_TOPOLOGY_REPLICATION_PASS` across GPT-2, distilgpt2, OPT-125M, and
SmolLM-135M. Each checkpoint passed its independent local reference,
symmetry, non-symmetry control, causal-atlas, and Jacobian-proxy gates. The
symmetry counts were 48/48, 24/24, 48/48, and 120/120, with all blind mappings
recovered. The control counts were 14, 8, 14, and 32; every control retained a
canonical digest difference. All four reports shared probe digest
`ef4e0b6ff2c8358f1e9350b06afae15c11d87b6bddc83e79cd6d14901f1035bf`.

The resulting claim ceiling is
`LOCAL_CROSS_TOPOLOGY_REPLICATION_OBSERVATION_ONLY`. This is reproducible
local evidence across the tested implementations and fixed probe contract. It
does not authorize provider calls, consume assessment data, establish
architecture-general proof or mechanistic understanding, validate alignment,
or establish production readiness.

Fresh-holdout SmolLM2-135M result: the pinned
`HuggingFaceTB/SmolLM2-135M` bundle at revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2` loaded from external owner-only
custody. Its eight artifact digests and 272 tensors were verified; the
checkpoint digest is
`c57149185c50e58bb323c58908720e8d1554d4e3ce6ff1fad804c964b2b58776`.
The checkpoint uses the same 30-layer Llama/GQA shape as SmolLM-135M but binds
RoPE base `100000` and context length `8192` from its own manifest.

The independent NumPy decoder matched local PyTorch/Transformers on the fresh
72-prompt suite with top-1 agreement `1.0`, mean relative L2 `1.36e-6`, and
95% bootstrap interval `[1.24e-6, 1.50e-6]`. The fresh suite digest is
`8b90443749e884dbd83a0f2e983c5445a1dd6439e8d79f9a5a13fcb88d1bd52b`, with
eight visible selection prompts and 64 held-out evaluation prompts.

SmolLM2 passed 120/120 symmetry cases, recovered 120/120 blind mappings, and
passed all 32 non-symmetry controls. Its atlas contains 90 gated-MLP entries
and 90 GQA-group entries; its local Jacobian proxy passed on layers 0, 14, and
29 with raw-coordinate change, inverse-aligned stability, and pruning
divergence. The model-specific claim ceiling remains
`LOCAL_SMOLLM_TRANSFORMATION_OBSERVATION_ONLY`.

The fresh-holdout result is evidence across two separately pinned
Llama-family model instances. It is not evidence that the criterion is
architecture-general, mechanistically explanatory, alignment-valid, or
production-ready.

Five-checkpoint fresh-holdout replay: the aggregate runner reported
`CROSS_TOPOLOGY_FRESH_HOLDOUT_REPLICATION_PASS`. GPT-2, distilgpt2, OPT-125M,
and SmolLM-135M all passed on the shared probe digest
`ef4e0b6ff2c8358f1e9350b06afae15c11d87b6bddc83e79cd6d14901f1035bf`.
SmolLM2-135M passed independently on fresh digest
`8b90443749e884dbd83a0f2e983c5445a1dd6439e8d79f9a5a13fcb88d1bd52b`.
The runner reports `same_probe_contract=false` intentionally, with
`shared_probe_contract=true` and `fresh_holdout_contract=true`; it never
pretends the fresh suite is the legacy suite. All five reports passed their
local reference, symmetry, control, atlas, and Jacobian-proxy gates.

The resulting ceiling is
`LOCAL_CROSS_TOPOLOGY_FRESH_HOLDOUT_OBSERVATION_ONLY`. This strengthens
replication across two Llama-family model instances but remains local,
checkpoint- and probe-contract-bound evidence. It does not establish
architecture-general proof, mechanistic understanding, alignment validation,
or production readiness.

Causal-prediction and activation-rescue result: the preregistered
`smollm2-causal-prediction-v1` protocol ran against the owner-custodied
SmolLM2-135M bundle in the repo-local Python 3.12.13 environment. The
runtime-bound checkpoint digest was
`b7a550fb21bc0b5a342f97a467ce2c554105919e0f35a75bc798f4dc9872d16d`; the
causal target-bound probe digest was
`f56f43ffdb106bd3a17bc40824299aef6db17286a80adbec5e9a3ba1738cd461`.
The model weights loaded from external custody; no provider was called and no
real assessment corpus was consumed.

Channel selection used only the eight visible probes and selected three
channels at each of layers 0, 14, and 29: `(898, 1297, 223)`, `(226, 75, 87)`,
and `(675, 385, 1233)`. The held-out set contained 64 probes. The mapping
comparison evaluated 576 MLP and 576 GQA-group observations. Canonical MLP
and GQA prediction error was `0.0` with 100% wins against both raw-coordinate
and fixed-seed random mappings. Raw-coordinate mean relative L2 was
`1.5952721936` for MLP and `5.8210934364` for GQA; fixed-seed random means
were `1.4276475126` and `8.1362257084`. The corresponding canonical 95%
bootstrap intervals were `[0.0, 0.0]`.

The operational alignment was blind to the recorded synthetic shift: each
selected layer/family mapping was inferred from exact composite component
signatures, then checked for raw-digest difference and canonical-digest
equality before it was used in the held-out predictions, patches, or rescues.

Activation patching evaluated 576 observations. The aligned canonical mean
relative L2 was `0.0`, with 95% interval `[0.0, 0.0]` and 100% wins against
raw patching; raw patching mean relative L2 was `12.0264040945`, with interval
`[6.26960088, 22.16632128]`. Full activation rescue evaluated 192
observations. Aligned rescue mean relative L2 was `3.298e-7`, interval
`[2.5e-7, 4.2e-7]`, and 100% of observations beat raw rescue; raw rescue mean
relative L2 was `0.5280996787`, interval `[0.45776913, 0.60728439]`.

The run reported `CAUSAL_PREDICTION_AND_RESCUE_PASS`. This is a stronger,
falsifiable local observation: for the tested SmolLM2 transformations, a
mapping inferred from component identity predicts held-out ablation effects
and restores held-out activation interventions, while raw and random mappings
do not. It remains an equivalence-class observation for one checkpoint,
architecture family, target-logit projection, and fixed probe contract. The
claim ceiling is
`LOCAL_CAUSAL_PREDICTION_AND_RESCUE_OBSERVATION_ONLY`; it is not mechanistic
understanding, architecture-general proof, alignment validation, independent
review, or production readiness.
