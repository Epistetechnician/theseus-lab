# Replacement-loss-v7: shape-safe downstream-aware replacement

State slice: `theseus-lab-sandbox-v1`.

Protocol ID: `theseus-replacement-loss-v7-downstream-aware-shape-safe`.

Status: preselection preregistration. Freeze this document, the v7 trainer,
fixture rules, compatibility runner, tests, dependency manifest, and pinned v3
and v6 implementation bytes before selecting a v7 checkpoint. v3 through v6
remain immutable. The v6 attempt is recorded as `INCONCLUSIVE`: its first OPT
case stopped during candidate fitting because the candidate projection output
was flattened while the captured target output was batch/sequence shaped. No
candidate freeze, held-out prediction, challenge evaluation, or outcome was
produced. v7 corrects that tensor-shape boundary and adds a regression test.

## Question and claim ceiling

Can the frozen replacement-loss-v6 procedure complete with a fresh affine
projection trained on disjoint adaptation prompts, and meet its existing
causal-prediction, behavior, activation, numerical-equivalence, and
incremental-baseline gates on nine post-freeze cases?

The candidate objective, thresholds, controls, model families, adaptation
counts, held-out challenge design, and resource limits remain those of v6.
Only the following are changed: hook-output normalization before activation
loss, v7 fixture namespaces and seeds, frozen v7 packet identity, and explicit
exclusion of every previously selected checkpoint identity. The optimizer
continues to use AdamW with learning rate `1e-3`, zero weight decay, and the
frozen eight-epoch schedule.

The v7 result can support only a local observation about this fitting
procedure and the exact selected checkpoints. It cannot establish an
architecture-general theorem beyond the additive-module class, general
mechanistic understanding, independent science, consciousness, alignment,
or production readiness. v6's incomplete attempt is not a measured failure
and is not evidence for or against the treatment.

## Targets, fixture partitions, and cases

After packet freeze, select exactly one public base checkpoint for each frozen
family: OPT/ReLU, GPT-2/GELU, and Llama/SiLU-gated. Require a full immutable
40-character source revision, at most 500 million parameters,
`trust_remote_code=False`, and the family's frozen MLP projection contract.
Reject all model identities selected in v3-v6, including
`facebook/opt-350m`, `openai-community/gpt2-medium`,
`HuggingFaceTB/SmolLM-360M`, `babylm/opt-125m-strict-2023`,
`BabyLM-community/babylm-baseline-100m-gpt2`, and
`babylm/babyllama-100m-2024`, as well as the earlier protocol exclusions.
If any family has no eligible fresh checkpoint, stop before candidate
preparation and report `NO_ELIGIBLE_FRESH_TARGET`.

Run seeds `20261201`, `20261202`, and `20261203`, yielding nine
checkpoint-by-seed cases. Use visible-only prompts to select the MLP output
projection with greatest mean per-feature input variance; tie-break by lower
layer index. The projections are OPT `fc2`, GPT-2 `mlp.c_proj`, and Llama
`mlp.down_proj`.

Each case uses 128 adaptation prompts: 64 ordinary, 32 compositional, and 32
adversarial. Each held-out challenge contains 60 fresh prompts: 20 per
stratum. v7 visible, adaptation, and challenge partitions must be disjoint by
example ID, exact prompt, and exact prompt-plus-continuation. They must also
be disjoint from every prior custodied v3-v6 fixture, including partial v6
visible and adaptation files. The v7 prefix and IDs are protocol-namespaced.
Prompt contents remain in external owner custody.

## Replacement, controls, and evaluation

Initialize a new `torch.nn.Linear` with the selected projection dimensions and
bias layout; copy no target weights. Freeze the base model. Train only the
replacement for eight epochs on adaptation prompts using CPU float32, AdamW,
zero weight decay, learning rate `1e-3`, and batch size 4. Initialize PyTorch
with the case seed. Each epoch uses a new row permutation from persistent
NumPy `default_rng(seed XOR 0x6A09E667)`. Do not early-stop or tune against
challenge data.

Normalize replacement hook outputs using the pinned v3
`_activation_bth` shape contract before computing activation loss. This accepts
both `[batch, sequence, hidden]` and flattened `[batch * sequence, hidden]`
outputs and rejects all other layouts. Preserve the raw replacement output at
the hook boundary so the target model consumes the expected tensor layout.

The treatment objective remains `L_activation + L_behavior`, with both
weights 1, next-token KL temperature 1, and activation loss masked over
non-padding prompt tokens. The output-only ridge comparator is fitted to the
same adaptation data. Preserve the pinned v3 random-target, raw-index
permutation, mapping-aligned permutation, and no-op controls. The historical
v3 wrong-module result remains separately reported; v6's evaluator slot alias
continues to mean output-only ridge for this incremental comparison.

After all nine candidates and receipts are frozen, draw fresh challenge
entropy. Seal treatment/control predictions before original-model held-out
outcomes are opened. Retain all v6 treatment gates and the primary paired
improvement metric and thresholds unchanged. Overall outcome is `PASS` only
when every frozen treatment gate and the incremental metric pass. Measured
misses are `FAIL`; missing artifacts, digest errors, incomplete prediction
sealing, or resource termination are `INCONCLUSIVE`.

## Custody, reproduction, and nonclaims

Use local CPU only. Retain model weights, tokenizer files, prompts, captures,
candidates, predictions, outcomes, and row-level diagnostics only under the
external v7 custody root. Repository files contain protocol code, frozen
checkpoint identities, and aggregate results. The packet binds exact v7
source, pinned v3/v6 dependencies, requirements, fixture rules, controls,
budgets, custody, claim ceiling, nonclaims, and stop conditions.

Run the exact frozen packet in the configured clean second runtime. Report
`REPRODUCED_LOCAL` only if candidate reconstruction, prediction and outcome
digests, source identity, dependencies, and distinct-runtime checks all match.
This same-owner reproduction is not independent science. External review
remains `EXTERNAL_REVIEW_UNAVAILABLE` until a configured reviewer returns a
signed decision binding the exact packet. Alignment, consciousness, and
production remain separately gated.
