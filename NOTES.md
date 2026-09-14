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
