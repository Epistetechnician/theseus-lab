# V36R1: custody repair and public checkpoint acquisition

Prepared 2026-10-06. Revision `202f459d6d89df9a038d78d58e0af825d287fcd8`
plus a digest-bound uncommitted packet. State slices:
`theseus-lab-sandbox-v1`, `theseus-pretrained-whole-block-transfer-v36r1`,
`theseus-public-checkpoint-acquisition-v36r1`,
`global-experiment-assurance-lifecycle-v1`.

## End state and lineage

Complete a newly frozen local whole-MLP transfer procedure using the
[V36 method](replacement-loss-v36-pretrained-transfer.md). V36 remains
`INCONCLUSIVE`: its strict historical collector encountered three legacy
Cloud Run exports that are not JSON. No checkpoint was selected, no target
was assessed, and no replacement was trained in V36. Its source is unchanged.

This fresh process adapter explicitly installs a new protocol ID, state slice,
artifact root, source freeze and intake functions. The original V36 capture,
fit, intervention, scoring and closeout source files are frozen members and
are reused without edits. The protocol ID changes corpus tags and deterministic
seed derivation. Replacement seeds remain 36101/36102/36103. Do not import this
adapter in a process that is executing the original V36 packet.

H1, the endpoint comparator, all numerical thresholds, two sequential MLP
replacements, three families, three seeds, adaptation/challenge counts, 1024
updates, four derivative probes, controls and prediction-sealing order are
identical to V36. No loss or observed-outcome tuning is introduced.
The claim ceiling remains `LOCAL_PRETRAINED_TWO_MLP_INTERFACE_REPLACEMENT_ONLY`.

## Historical intake repair

The full prior JSON inventory is scanned once. Every valid JSON record retains
its path, byte count, digest and identity extraction. Three exact legacy paths
are quarantined by the source-coded byte/SHA-256 allowlist: the 550-byte final
Cloud Run log, the empty progress export, and the empty task-start export.
Their original bytes remain unchanged. Changed or missing allowlisted bytes,
any other malformed JSON, oversized input or escaping link stop the new packet.

Quarantined files participate in custody binding but supply no identity rows.
This does not establish absence from opaque, unretained or non-JSON-only
predecessor corpora. No stronger global disjointness claim is made than the
retained JSON identity inventory plus known checkpoint registry. The intake
and frozen source archives are verified before acquisition and each study run.

## Explicit public asset adapter

V36 prohibited downloads. This fresh packet adds an opt-in static public-model
adapter under `theseus-public-checkpoint-acquisition-v36r1`, exercising the
user's authorization to complete fresh-checkpoint research locally. It does
not enable inference APIs, paid compute, cloud workers, external review,
production traffic, deployment, publication, commit or merge.

Acquire requires `--execute-public-acquisition`, a verified source freeze and
intake, mounted custody, and a fresh acquisition root. The endpoint is fixed
to `https://huggingface.co`; every client call passes `token=False`. Credentials
are neither used nor recorded. API behavior follows the inspected pinned
Hub 1.31.0 implementation and [official API documentation](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api).

Metadata queries are fixed: `opt`, `gpt2`, `llama`, with Transformer
text-generation filters, parameter maximum 250M, creation-time sort, and at
most 200 catalog entries per family. Within each captured catalog, lexicographic
model ID order nominates the first metadata-qualified entry. No generated
response, activation, removal score or held-out result participates in selection.

Eligibility requires an unseen recorded identity; a supported architecture and
activation; width 8–1024, depth≥4, vocabulary≤65,536, parameters≤250M; safe
weights and a tokenizer; public ungated access; a full 40-hex revision; and
≤1.1 GiB selected files/model. Reject named random/dummy/instruction/chat/
fine-tuned/LoRA/quantized entries, nonempty model-card base-model references,
unknown licenses, and absent explicit from-scratch training provenance.
The pinned README must declare `(pre-trained|pretrained|trained) from scratch`.

Declared license set: Apache-2.0, MIT, BSD-2/3-Clause, CC0-1.0, CC-BY-4.0.
License and training provenance are publisher/custodian evidence, not independent
authority. The card, config and weight-file digests bind nominations. No custom
Python or pickle weight file is acquired or executed. Meta-device skeletons
count parameters using installed trusted Transformers code; no learned weight
or model output is queried for metadata qualification.

All three nominations freeze before downloading any candidate weight. If a
family has no metadata-qualified candidate in the declared catalog, stop
`NO_ELIGIBLE_CHECKPOINT_COHORT`, retain the screen and acquire no weights.
There is no second cohort after weight, qualification, fitting or held-out faults.
Candidates are copied only into this packet's external custody. SHA-256 weight
binding detects exact artifact reuse; it is not a complete canonical-parameter
audit of differently serialized aliases. Pinned from-scratch provenance provides
an additional check but is not independent validation.

Acquisition budget: 1800 seconds and 8 GiB retained acquisition bytes,
cooperatively checked between calls. Metadata config/README files≤2 MiB each;
selected downloads use two workers. Library HTTP retries are not additional
scientific attempts; transport failure terminates the acquisition root.
Network traffic including retries is unmeasured. No prompts or outcomes leave
local custody; public metadata requests disclose public search/repository IDs.

## Runtimes and scientific execution

Artifacts:
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-pretrained-whole-block-transfer-v36r1`.
Runtimes:
`/Volumes/Backup ED/ResearchRuntimes/theseus-lab-v36r1/toolchains`.
All instantiated data, candidate weights, predictions, oracle/rescue arrays,
public assets and detailed receipts stay external. Repo contains only source,
rules, tests, freezes and scalar reports.

Bootstrap creates two isolated Python 3.12.13 environments from exact installed
dependency files in the V36 34-package lock, excluding bytecode and external
entrypoint shims. Every copied file's size/digest is checked; copies are limited
to 1 GiB combined. Base interpreter and system libraries are shared on one host.
Record positive runtime checks, separate prefixes/site-packages, and interpreter/
dependency-file hashes before model execution. Same-owner reproduction can
reach only `REPRODUCED_LOCAL`, never independent acceptance.

The scientific runner retains V36's one-thread float32 CPU/eager/deterministic
settings, 24-hour per-runtime wall cap, 6 GiB peak RSS, 24 GiB artifacts and
25 GiB pre-fit free-space guard. Monetary cap $0. Paid execution remains closed;
energy, amortization and network economics are unmeasured. Resource caps are
cooperative, not hard preemption. Preserve all actual failures literally.

Visible qualification misses stop `NO_QUALIFIED_TARGET` before training. Study
faults are `INCONCLUSIVE`; retention misses are `FAIL`. An all-nine treatment
`PASS` must also satisfy every stage/stratum and control. Replica reconstruction,
prediction sealing, outcome equality and separate numeric closeout are required
for an actual `REPRODUCED_LOCAL` finding. No result is presumed here.

Commands after source freeze:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python replacement_loss_v36r1_runner.py verify
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python replacement_loss_v36r1_runner.py intake
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python replacement_loss_v36r1_runner.py acquire --execute-public-acquisition
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python replacement_loss_v36r1_runner.py bootstrap
PYTHONDONTWRITEBYTECODE=1 <primary-python> replacement_loss_v36r1_runner.py runtime-check
PYTHONDONTWRITEBYTECODE=1 <replica-python> replacement_loss_v36r1_runner.py runtime-check
PYTHONDONTWRITEBYTECODE=1 <primary-python> replacement_loss_v36r1_runner.py run --role primary
PYTHONDONTWRITEBYTECODE=1 <replica-python> replacement_loss_v36r1_runner.py run --role replica
PYTHONDONTWRITEBYTECODE=1 <locked-python> replacement_loss_v36r1_runner.py closeout
```

Independent science, novelty priority, mapping equivalence, metaphysical identity,
consciousness, human alignment and production remain unestablished. Reviewer
`NOT_SUBMITTED`; the next authority gate for stronger scientific claims remains
the separately authorized configured independent reviewer.
