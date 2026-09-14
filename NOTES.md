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

The report claim ceiling is now
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
coordinates recover the baseline and pruning diverges. The claim ceiling remains
`LOCAL_GPT2_TRANSFORMATION_OBSERVATION_ONLY` plus
`LOCAL_JACOBIAN_PROXY_OBSERVATION_ONLY`; no general mechanistic, scientific,
alignment, or production claim follows.
