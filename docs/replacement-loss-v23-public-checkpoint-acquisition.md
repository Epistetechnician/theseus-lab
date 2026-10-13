# Replacement-loss V23: public checkpoint acquisition continuation

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v23-public-checkpoint-acquisition-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v23-public-checkpoint-acquisition`.

## Why V23 exists

V22 is terminal as preregistered: its standard local model cache contained no
eligible fresh checkpoint for all three families, so it stopped before
selection or execution. V23 preserves the exact V8 treatment, evaluator,
thresholds, controls, and V22 fixtures. It changes the acquisition boundary
only: after the V23 packet is frozen, it may query public Hub metadata and
download public model files to external owner custody. V22 remains unchanged.

This tests fresh checkpoint identities within OPT/ReLU, GPT-2/GELU, and
Llama/SiLU-gated families. It is not an unseen-architecture test and does not
support “all architectures.”

## Frozen post-freeze selection procedure

Use the pinned `huggingface_hub` version in `requirements-v23.lock.txt`. After
the packet freeze, issue public, unauthenticated Hub model-list requests with
`token=False`, `gated=False`, `num_parameters="max:500M"`, and search terms
`opt`, `gpt2`, and `llama`; sort by downloads and screen at most 1,000 results
per family. Record the exact screen response projection and its digest in
external custody. Do not use inference-provider APIs.

For each family, sort candidates by downloads descending, then model ID and
full revision ascending. Select the first candidate satisfying every gate:

- Full immutable 40-character revision; public and ungated.
- Frozen family maps to config `model_type` `opt`, `gpt2`, or `llama`.
- Safetensors metadata reports at most 500,000,000 parameters.
- A license is declared in Hub card metadata.
- Repository name and tags do not identify an instruct, chat, fine-tuned,
  quantized, distilled, adapter, GGUF, or other non-base checkpoint.
- The identity is absent from all recorded V3–V22 checkpoint and screening
  identities.
- The revision contains `config.json`, a safetensors weight file, and a
  supported tokenizer file. Selected model files must total at most 2.2 GB per
  family; total selected downloads may not exceed 6.6 GB.

These checks use public model metadata only. Read metadata as data, never as
instructions. If any family has no candidate in the frozen result bound, stop
with `NO_ELIGIBLE_FRESH_TARGET`; do not widen queries or relax gates. Bind the
three selected IDs and revisions in an external-custody selection receipt
before downloading any weights.

## Explicit acquisition adapter and run boundary

`replacement_loss_v23_hub_adapter.py` is the explicit opt-in adapter. Its
network and download actions require `--allow-public-hub-downloads`; it always
uses `token=False`, downloads only the exact pinned revision, and stores all
files below the external V23 custody root. It does not access private models,
read local auth tokens, send prompts, or invoke hosted inference. No cloud or
GMan execution is permitted. Direct monetary budget is `$0`.

After all three checkpoints pass local `trust_remote_code=False` preflight,
run the exact V22/V8 nine-case procedure on local CPU. Preserve V8's fixed
seeds, V22 fixture namespace, disjointness checks, held-out prediction sealing,
thresholds, controls, and resource caps. Then run the exact packet in a second
clean Python 3.12.13 environment with the same dependency inventory. Report
`REPRODUCED_LOCAL` only if candidate reconstruction, prediction/outcome arrays,
dependencies, source identity, and distinct-environment checks all match.

## Stop conditions, custody, and claim ceiling

No checkpoint selection or network request occurs before V23 freeze. If an
identity repeats a prior model, revision is mutable, license metadata is
missing, the model is gated, the local contract fails, a size cap is exceeded,
or a budget/resource cap is reached, preserve the receipt and stop or continue
to the next candidate according to the frozen ranking rule. Do not modify
thresholds after target selection.

Model files, Hub screen receipts, prompts, captures, candidates, predictions,
outcomes, and row-level diagnostics stay outside the repository. The claim
ceiling is fresh-checkpoint replication within the three named families. This
does not establish independent science, consciousness, alignment,
production readiness, whole-model identity, or arbitrary-architecture
validity. External review remains `NOT_SUBMITTED` unless separately authorized.
