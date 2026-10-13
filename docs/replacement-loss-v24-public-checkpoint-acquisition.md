# Replacement-loss V24: client-compatible public checkpoint acquisition

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v24-public-checkpoint-acquisition-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v24-public-checkpoint-acquisition`.

## Why V24 exists

V23 froze the right public checkpoint criteria, but its pinned
`huggingface_hub==0.36.2` client does not accept the newer `num_parameters`
argument. V23 stopped before any model records were returned. V24 preserves the
V8 treatment/evaluator and V22 fixtures, but performs the exact 500M parameter
check locally from safetensors metadata after a bounded Hub search supported by
the pinned client. V23 remains unchanged.

This is a fresh-checkpoint replication within OPT/ReLU, GPT-2/GELU, and
Llama/SiLU-gated families. It is not an unseen-architecture test.

## Frozen Hub screen and selection

After the V24 packet is frozen, use `huggingface_hub==0.36.2`, public
unauthenticated requests (`token=False`), `gated=False`, downloads-descending
order, and at most 1,000 results per query. Query `opt` for OPT, `gpt2` for
GPT-2, and `llama`, `smollm`, `open-llama`, and `tinyllama` for the Llama family.
Deduplicate each family by `(model_id, full revision)` and sort the union by
downloads descending, then model ID and full revision ascending. The search
does not send prompts and does not call any inference endpoint.

Screen candidates in that order. The first candidate per family must be public
and ungated, have a full 40-character commit, map to the frozen `model_type`
(`opt`, `gpt2`, or `llama`), have a declared license, have no frozen name/tag
markers for instruction/chat/fine-tuning/quantization/distillation/adapters,
contain supported config, tokenizer, and safetensors files, and have an exact
safetensors parameter count no greater than 500,000,000. Reject identities in
the V3–V23 use and screening records. Selected model files may total at most
2.2 GB per family and 6.6 GB overall. Persist all query projections,
rejections, selected identities, revisions, license metadata, and file hashes
only in external custody.

If any family has no eligible candidate in the frozen result bound, stop with
`NO_ELIGIBLE_FRESH_TARGET`; do not widen queries or relax gates. If all three
qualify, bind the selected identities before downloading exact revisions.

## Acquisition, execution, and assurance

The explicit opt-in adapter requires `--allow-public-hub-downloads`; it sets
`token=False` and stores only allowlisted config, safetensors, and tokenizer
files below the external V24 custody root. Before importing Hub clients, V24
redirects `HF_HOME`, `HF_HUB_CACHE`, `HF_ASSETS_CACHE`, and `HF_XET_CACHE` to
subdirectories of that root and disables Hub telemetry. This also contains
temporary safetensors-index downloads made by the pinned metadata checker. It
does not access local auth tokens, gated models, or hosted inference. No
cloud/GMan workers are used. Direct monetary budget is `$0`.

Run the exact V22/V8 nine-case method on local CPU: same V22 fixture namespace,
seeds, disjointness checks, candidate sealing, interventions, controls,
thresholds, and resource caps. If primary evaluation completes, run a second
clean Python 3.12.13 environment with matching dependency inventory. The
primary and clean-replica artifact roots are pinned under
`/Volumes/PrimaryED/ResearchArtifacts` by the frozen runner. The complete
26-package environment is pinned in `requirements-v24.lock.txt` and its
inventory digest is bound by the freeze manifest.
`REPRODUCED_LOCAL` requires exact candidate reconstruction, prediction/outcome
digests, dependency equality, source identity, and distinct environment. A
reproducible `FAIL` remains a `FAIL` experiment outcome.

The runner permits one metadata screen, one primary evaluation, and one clean
reproduction per frozen custody root. API, download, evaluation, or
reproduction exceptions produce a digest-bound stop receipt and preserve any
partial external-custody outputs. The runner does not retry or retune that
packet; a later attempt requires a new protocol version. The V8 report
separates behavior, causal-effect, and activation-rescue metrics but does not
define a separate behavior-versus-mechanism divergence statistic, so V24 will
report that diagnostic as not formally defined by the frozen method.

Keep model files, prompts, captures, candidates, predictions, outcomes, and
row-level diagnostics outside the repository. Reviewer status is
`NOT_SUBMITTED`. The claim ceiling is fresh-checkpoint replication within the
three named families. This does not establish unseen-architecture validity,
whole-model identity, independent science, consciousness, alignment, or
production readiness.
