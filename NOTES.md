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
logit error `4.8e-7` across 16 prompts. MLP-channel and attention-head
permutations passed at layers 0, 5, and 11. Every case preserved visible and
held-out top-1 behavior, changed 100% of the selected parameter positions, and
restored intervention effects under the inverse mapping with cosine at least
`0.999` and maximum absolute logit difference below `1e-4`.

The final-layer MLP case has aligned-intervention relative error `0.0192734`
because the intervention vector is small; its maximum absolute difference is
`3.815e-5` and cosine is `0.9998647`. Relative error remains reported as a
diagnostic; the acceptance rule uses the predeclared cosine and absolute-error
limits to avoid treating small-vector floating-point amplification as causal
divergence.

Non-symmetry result on the expanded 16-prompt set: single-channel ablation at
layers 0 and 5, block pruning of 128 layer-5 channels, and layer-5 channel
replacement all produced measurable logit divergence. The single-channel
ablation at layer 11 did not cross the `1e-4` relative-logit threshold and did
not change top-1 outputs; this is a probe-bounded sensitivity result, not proof
that the channel is globally irrelevant. Block pruning changed top-1 output on
1 of 8 visible prompts while preserving all held-out top-1 outputs, showing that
logit and top-1 metrics can disagree.

The report claim ceiling is now
`LOCAL_GPT2_TRANSFORMATION_OBSERVATION_ONLY`. No general mechanistic,
scientific, alignment, or production claim follows from this prototype.

Formalized identity criterion result: all six selected symmetry cases (MLP
channels and attention heads at layers 0, 5, and 11) passed the exact four-part
criterion: raw digest difference, canonical digest equality after inverse
mapping, visible plus held-out behavior match, and aligned intervention match.
The canonicalization is an explicit invariant for the recorded permutation;
it is not a learned similarity metric. The five ablation, pruning, and
replacement controls retain canonical digest differences and are never
accepted as identity-equivalent.

The expanded causal atlas contains 18 entries: three MLP channels and three
attention heads at each of layers 0, 5, and 11. Every entry reports mean
relative logit effect and a deterministic 95% bootstrap interval over the 16
fixed probes. The selected MLP-channel effects ranged from `2.22e-6` to
`6.7972e-4`; selected attention-head effects ranged from `0.00406515` to
`0.08473632`. Top-1 changes were sparse and sometimes absent despite a
nonzero logit effect, so top-1 alone is not treated as causal evidence.

The local Jacobian-lens proxy fits on the eight visible prompts and evaluates
on the eight held-out prompts. Its target tokens are `(" the", " is", " of",
" and")`, IDs `(262, 318, 286, 290)`, with target digest
`8c055c3ea9637aa48eeab068e2411adeeccf9180a04eb25f7e8c135583346ae6`. The
permutation stability check passed at all three layers. Mean held-out
coordinate relative error was `1.0e-7`, `1.2e-7`, and `2.58e-6` at layers 0,
5, and 11; the corresponding 95% intervals were `[6e-8, 1.4e-7]`,
`[9e-8, 1.7e-7]`, and `[8.6e-7, 5.54e-6]`. Pruning 128 channels diverged in
the proxy at every selected layer, with mean relative errors `0.00864998`,
`0.01526452`, and `0.11627851`.

The resulting insight is narrower and falsifiable: within this checkpoint and
probe contract, channel/head labels are not identity primitives; identity is
better modeled as an equivalence class under validated function-preserving
symmetries plus aligned causal interventions. The Jacobian proxy is
consistent with that criterion, but its stability is not independent evidence
of mechanistic understanding because the tested permutation preserves the
post-block residual exactly. The claim ceiling remains
`LOCAL_GPT2_TRANSFORMATION_OBSERVATION_ONLY` plus
`LOCAL_JACOBIAN_PROXY_OBSERVATION_ONLY`; no general mechanistic, scientific,
alignment, or production claim follows.
