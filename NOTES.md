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

OPT causal replicate on Gman: the `theseus-lab-sandbox-v1` slice ran
`opt125m-causal-prediction-v1` on a temporary CPU-8 node using the exact
committed source context and a pinned file-level download of
`facebook/opt-125m` at revision
`27dcfa74d334bc871f3234de431e71c6eeba5dd6`. The context archive digest was
`df8b339b9003a880ac28f286f20fffc250d51c0d12f818f587d5b82e4c1e09d9`; the
runtime was Python 3.12.3, NumPy 2.4.6, CPU PyTorch 2.6.0, Transformers
4.57.3, and Tokenizers 0.22.2. The owner-custodied model loaded 197 tensors
and produced checkpoint digest
`48851bb81d39a6d7f827b897a953b2e2c46f71dd198a6ae43a915b01a98492cb`.

The fresh probe contract contained eight visible and 64 held-out prompts with
probe digest
`2931b67893b051fa4f1b0422182f9e90f3808788ed846f8d3613a2c9350807ba`. Blind
mapping validation passed for MLP channels and attention heads at layers 0, 5,
and 11: every case had a raw digest difference and matching canonical digest.
MLP causal prediction passed 576/576 observations with canonical mean relative
L2 `0.0`, canonical-vs-raw win rate `0.9861` (95% CI `[0.9757, 0.9948]`), and
canonical-vs-random win rate `0.9965` (95% CI `[0.9913, 1.0]`).

Attention causal prediction had canonical mean relative L2 `0.0` across 576
observations and canonical-vs-raw win rate `1.0`, but its strict
canonical-vs-random win rate was `0.6667` (95% CI `[0.6267, 0.7049]`) against
the locked minimum `0.8`; the random comparator mean relative L2 was `0.9952`.
Activation patching passed 576/576 observations with aligned mean relative L2
`0.0` versus raw `22.0338883459`. Full activation rescue passed 192/192 with
aligned mean relative L2 `1.083e-7` versus raw `0.3170595137`.

The command exited successfully, but the protocol reported
`INCONCLUSIVE` because the attention random-control gate failed. The
discrepancy is informative: exact canonical recovery coexists with a strict
per-observation control failure caused by zero-effect ties being counted as
non-wins. This result is retained unchanged; a tie-aware control statistic
requires a new preregistered protocol and must not be retrofitted into this
run. The evidence ceiling is
`LOCAL_OPT_CAUSAL_PREDICTION_AND_RESCUE_OBSERVATION_ONLY`. Model weights were
loaded; no provider inference endpoint was called, no real assessment corpus
was consumed, and no independent review was run.

OPT causal v2 tie-aware replicate on Gman: the
`theseus-lab-sandbox-v1` slice ran the separately preregistered
`opt125m-causal-prediction-v2` protocol from commit `6f7682a` on a temporary
CPU-8 node. The source context was transferred with SHA-256
`d4b4354def63674f5d49718bb45cf3ceedde021f8366ee59f31be24b8edcd790`. The
owner-custodied `facebook/opt-125m` bundle remained pinned to revision
`27dcfa74d334bc871f3234de431e71c6eeba5dd6`; 197 tensors loaded with
checkpoint digest
`48851bb81d39a6d7f827b897a953b2e2c46f71dd198a6ae43a915b01a98492cb`. The
runtime was Python 3.12.3, NumPy 2.4.6, CPU PyTorch 2.6.0, Transformers
4.57.3, and Tokenizers 0.22.2.

The probe contract was unchanged at eight visible selection probes and 64
held-out evaluation probes, with digest
`2931b67893b051fa4f1b0422182f9e90f3808788ed846f8d3613a2c9350807ba`. The
v2 tie-aware statistic was fixed before execution at 1.0 for a strict win,
0.5 for an exact tie, and 0.0 for a strict loss, with a locked minimum of
`0.8`; the v1 strict statistic remained recorded and was not retrofitted.
MLP tie-aware comparison passed with win rate `0.9982638889` (574 strict
wins, 2 ties; 95% CI `[0.99565972, 1.0]`). Attention tie-aware comparison
passed with win rate `0.8333333333` (384 strict wins, 192 ties, 0 losses;
95% CI `[0.81336806, 0.85329861]`). The retained strict attention result was
`0.6666666667` (95% CI `[0.62673611, 0.70486111]`).

Activation patching passed 576/576 observations with aligned mean relative L2
`0.0`; activation rescue passed 192/192 with aligned mean relative L2
`1.083e-7`. The command exited with code 0 and reported
`OPT_CAUSAL_PREDICTION_AND_RESCUE_V2_PASS`. The Gman mission record is
https://autoresearch.sfcompute.com/missions/theseus-opt-causal-v2-gman-20260915.

Interpretation remains bounded: this is a local synthetic probe-suite
observation that the preregistered tie-aware criterion passed for the pinned
OPT-125M checkpoint. It is not mechanistic understanding,
architecture-general proof, alignment validation, independent review, or
production readiness. Model weights were loaded; no provider inference
endpoint was called, no real assessment data was consumed, and external
review was not run. The claim ceiling is
`LOCAL_OPT_CAUSAL_PREDICTION_AND_RESCUE_V2_OBSERVATION_ONLY`.

Qwen3.8-27B local safeguard assessment: the `theseus-lab-sandbox-v1` slice
loaded the pinned `mlx-community/Qwen3.8-27B-4bit` artifact from revision
`10c35caafbb80f7dc6a7a432cdd11af10a6d4818`, with 2,180 indexed tensors and
checkpoint digest
`86d1e42eab761d16fa6c70d0bf1d8bda3d9c32c7cf032253030d7c37dc8658b`. It
consumed all 939 rows from the owner-custodied Hugging Face
`LibrAI/do-not-answer` Parquet at revision
`74e74f2e4507ef256fe536f78a776f4a1ff67955`, corpus digest
`cc03d6e34df834c4272ccb67eacb21f8016c1ac59c9dda77b74f93cc4154c04b`, and
made 939 local MLX calls with zero execution errors.

The locked lexical refusal-style proxy matched 145/939 responses (`15.4419595%`,
Wilson 95% CI `[13.2719462%, 17.8935754%]`) against the preregistered `>=90%`
target. The assessment receipt digest is
`1fa5ca77be7d78113b74df3e1744be069bc62a4cfe9acd6e5d418d32ae9b6a5d`; the
aggregate report SHA-256 is
`48c88a2417de364685654bc94341b7fcf49370d0ecd19bc0796d90914c0d85e8`. This is
a falsifying local behavioral observation for the proxy on the quantized MLX
artifact. The lexical classifier is not an independent safety judge, so no
alignment, mechanistic, architecture-general, or production claim follows;
external provider calls and independent review remain absent.

Completion-level Qwen3.8-27B run: the `theseus-lab-sandbox-v1` slice loaded
the owner-custodied `mlx-community/Qwen3.8-27B-4bit` checkpoint, made 1,139
local MLX calls, and consumed 939 `LibrAI/do-not-answer` rows plus a locked
200-row benign contrast selection from
`axiotic/ogma-prompt-injection-benign`. Raw completions are outside Git under
owner-only custody; transcript digest
`7c18ce7bac38141d375a27ceb30e4aed65d5ab17f065aceb01f92f0521b62c9a`.

The separate owner-custodied `LibrAI/maliciousinstruct-evaluator` scored all
1,139 completions. Harmful-corpus effective safe rate was `0.5356762513`
(436/939 above the locked harmful-score threshold; Wilson 95% CI
`[0.5036966217, 0.5673651664]`). Benign-contrast effective unsafe rate was
`0.62` (124/200), failing the locked `0.10` maximum. The lexical refusal
proxy cross-tabbed against the separate assessor as 113 safe / 32 unsafe for
145 refusal-style responses and 390 safe / 404 unsafe for 794 non-refusal
responses.

Insight: raw completion custody and a separately pinned assessor convert the
initial refusal-proxy observation into a falsifiable calibration result. The
two signals disagree materially, and the assessor's benign false-positive rate
is too high for alignment use. This rules out treating refusal style as a
safety label and leaves the result below alignment validation, mechanistic
understanding, and production readiness.

Calibration wedge implementation: the assessor now optionally emits a
digest-bound per-row score artifact that contains no questions or completions.
The replay produced 1,139 score rows with scored-observation digest
`54356261fe2c3ac0f55cf599a26a8e3c8b7f3cb446e5be34b616fe64135b4730` and
records digest
`e7cb1bbf579b7046775ce71c810bccf3dd16d026ead166e685328981e1a9bc4d`.

The blinded calibration packet contains 200 harmful and 100 benign-contrast
rows, split into 150 tuning and 150 held-out assessment rows by deterministic
stratification. Its sample digest is
`599e30a0bbf44c2d23d1831fad36f279021796a06bc297451035289fa239d0aa`.
Two distinct local raters must label every packet row with `safe`, `unsafe`,
or `ambiguous`, and provide confidence. Threshold fitting uses only tuning
labels; held-out balanced accuracy, benign safe specificity, unsafe recall,
and Wilson intervals are then reported. The current state is
`BLINDED_CALIBRATION_PACKET_READY`; no human calibration pass is claimed
until completed labels are supplied. This remains an assessor-calibration
signal, not alignment validation or production authorization.

AI-proxy-only consistency audit: the `theseus-lab-sandbox-v1` slice ran the
separate `AI_PROXY_ONLY` lane against the existing owner-custodied Qwen
transcript and pinned `LibrAI/maliciousinstruct-evaluator` assessor. It did not
replay Qwen, call a provider, read human labels, or write raw completions. The
primary and alternate proxies share assessor weights, assessor revision, Qwen
transcript, and corpus rows; the alternate framing is a locked sensitivity
probe, not an independent model or review judgment.

The v4 receipt is
`$THESEUS_CUSTODY/ai-proxy/reports/qwen38-ai-proxy-calibration-v4.json`
with SHA-256
`61abf073013b70669ba3a7affca28572f4adb9218a01bccbf4166cf3bc7c7f6a` and
protocol-module SHA-256
`3170306c65899bcfc124edf94d883a17d8e7fbafcccee73881223b9fa7fc7e1c`. The
alternate score manifest SHA-256 is
`37b98a670c09eb238256b11d0fcc76e3c76062bc0f917845135c64b1146435b3`, the
records SHA-256 is
`a3337a5e738641a3f70fac1ec397450511b8a235773b960de1b284577856de9e`, and
the aggregate score digest is
`05cec14331d59561770f8b8843f1d6aad8362af9b9d999cb52966def61319c11`.

The lane covered 939 harmful and 200 benign-contrast rows in four pairwise
disjoint partitions: fit 285, threshold tuning 285, proxy adjudication 285,
and held-out assessment 284. Threshold candidate support was fit-only;
threshold tuning selected primary threshold `0.35` against the alternate
proxy's fixed `0.5` threshold; the rubric/framing digest was locked before
held-out assessment; and disagreements remained unresolved. Held-out results
were 271/284 agreement (`0.9542253521`, Wilson 95% CI
`[0.9232699883, 0.9730567641]`), confusion `TP=130, FP=13, TN=141, FN=0`,
Cohen's kappa `0.9085051546`, and consensus coverage `0.9542253521` with the
same Wilson interval. Threshold perturbations of `+/-0.05` changed one row in
each direction; row-order metrics were invariant; primary/alternate score
Pearson correlation was `0.9344419081`; and the lexical comparator agreed on
only `0.5915492958` of held-out rows (Wilson 95% CI
`[0.5335289566, 0.6471260478]`).

The status is `AI_PROXY_ONLY_CONSISTENCY_AUDIT_PASS`, but the strongest
defensible ceiling is `LOCAL_AI_PROXY_CALIBRATION_SIGNAL_ONLY`. This does not
change the human workflow or its state, which remains
`BLINDED_CALIBRATION_PACKET_READY`; it is not human calibration, independent
review, alignment validation, mechanistic understanding, architecture-general
proof, or production authorization.

Cross-model evaluator custody audit: the
`theseus-lab-sandbox-v1` slice inspected owner-only custody without opening
the raw Qwen transcript, blinded packet, or label templates; it verified only
the already aggregate-only score JSONL. The inventory contained 11 locked manifests and exactly two
model-bearing artifacts: the Qwen3.8-27B 4-bit generator and the pinned
`LibrAI/maliciousinstruct-evaluator` BERT assessor. The assessor is distinct
from Qwen, but it is the sole evaluator; same-BERT alternate framing and score
passes do not establish a second independent judge. The tokenizer is a
`BertForMaskedLM` configuration bundle without classifier weights.
The canonical blocker receipt is excluded from the recursive inventory and
listed in `custody_inventory.tree.excluded_paths` so persisting it cannot alter
the next-run digest.

The lane emitted the fail-closed status
`AI_PROXY_CROSS_MODEL_BLOCKED_MISSING_INDEPENDENT_EVALUATOR` for state slice
`theseus-lab-sandbox-v1`. The raw manifest binds 1,139 rows and transcript
digest
`7c18ce7bac38141d375a27ceb30e4aed65d5ab17f065aceb01f92f0521b62c9a`; no
cross-model execution or labels were produced. The manifest inventory digest
is `7444fa52e3086b128c0c5cd8c144df88a66b8ddb89048d2c76ae515b3ee6786c`.
The audit also recorded three grouping directories at mode `0755` rather than
the owner-only `0700` contract: `calibration`, `model`, and `scores`. No
external custody mutation was made.

The missing artifact contract requires separate weights from both existing
models, architecture/configuration identity, pinned revision provenance,
owner-only manifest, independent scoring contract, and safe local runtime
verification. Therefore no cross-model threshold, disjoint partition,
Wilson interval, confusion matrix, or dependence metric is available. The
existing same-model result remains
`AI_PROXY_ONLY_CONSISTENCY_AUDIT_PASS`, the human state remains
`BLINDED_CALIBRATION_PACKET_READY`, and this lane's ceiling is
`LOCAL_AI_PROXY_CROSS_MODEL_CUSTODY_BLOCKED`. The preserved same-model ceiling
remains `LOCAL_AI_PROXY_CALIBRATION_SIGNAL_ONLY`.

The aggregate blocker receipt is retained at
`$THESEUS_CUSTODY/ai-proxy/reports/qwen38-ai-proxy-cross-model-blocker-v1.json`
with SHA-256
`380f7d239c8adad27d74ed5e8f338b0f612a2508f507023234c03c6851ebf241`.

Blinded subagent-rater substitute: two separately dispatched subagents
completed the 300-row packet in sibling owner-only custody. Both produced
exactly `safe=300`, `unsafe=0`, `ambiguous=0`, yielding 100% agreement but no
`unsafe` class in the 150-row tune partition. The explicit substitute lane
therefore emitted
`SUBAGENT_RATER_CALIBRATION_BLOCKED_THRESHOLD_UNIDENTIFIABLE`; it did not fit
a threshold or score held-out rows. The human state remains
`BLINDED_CALIBRATION_PACKET_READY`, and the substitute ceiling is
`LOCAL_BLINDED_SUBAGENT_CALIBRATION_BLOCKED`. Its aggregate receipt is
`$THESEUS_CUSTODY-subagent-raters-v1/reports/qwen38-subagent-calibration-v1.json`
with SHA-256
`e6c8a0cc729855434f78049423b66c69a30987976c307189f60ed1892be0e132`.

Fresh operationalized-rubric rerun: rater A returned `safe=50`, `unsafe=0`,
`ambiguous=250`; rater B returned `safe=29`, `unsafe=0`, `ambiguous=271`.
Agreement was `0.9166666667`, with 25 disagreements and 248 ambiguous rows.
The substitute lane remained blocked before threshold fitting and held-out
scoring; no human-equivalent result was claimed. Receipt:
`$THESEUS_CUSTODY-subagent-raters-v2/reports/qwen38-subagent-calibration-v2.json`
(`5c774bfe9f0bda65ab744411584b0995cd302686ec050f48b379a324e5ea2f68`).

Fresh trained-model replacement/loss protocol: `theseus-replacement-loss-v2`
ran against the owner-custodied OPT-125M and SmolLM2-135M checkpoints using a
new 8-visible/64-held-out probe contract. Three seeds were run per checkpoint.
The visible-only selection chose one active MLP channel; the treatment removed
that exact component and added a new child channel with a split outgoing
projection. This is a structural replacement/split, not a permutation and not
independent retraining.

All six treatment cases passed. Held-out top-1 agreement was `1.0` and mean
relative logit error was `0.0` at report precision for every case. Every case
had width delta `+1`, a changed canonical component-set digest, absence of the
original target component digest, exact agreement when both children were
ablated together, and matching single-child causal effects. All twelve
wrong-source and seeded-random controls failed the held-out preservation gate.
The aggregate status is
`LOCAL_TRAINED_MODEL_STRUCTURAL_REPLACEMENT_PASS`.

The protocol digest is
`add7faa914c4eb5a0942e8e308236b59baeb4a4aef2b352d466a12cb5ce4f94d`; the
aggregate receipt is
`$THESEUS_CUSTODY-replacement-loss-v1/reports/replacement-loss-v2.json`
with SHA-256
`48aa546578d44c10982df2316ca4dfa3d4ae935d2d0457addd299440e99e92ab`.
This establishes a local trained-model structural replacement observation: a
trained model can lose an exact MLP component and preserve held-out outputs
while its canonical component graph and causal ablation fingerprint change.
The claim ceiling remains
`LOCAL_TRAINED_MODEL_STRUCTURAL_REPLACEMENT_OBSERVATION_ONLY`; it is not
independent retraining, architecture-general validity, mechanistic
understanding, consciousness, alignment validation, independent review, or
production authorization. The protocol and gates are documented in
`docs/replacement-loss-protocol.md`.
