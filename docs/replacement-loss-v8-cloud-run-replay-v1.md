# Replacement-loss v8 Cloud Run replay

State slices: `theseus-lab-sandbox-v1` and
`global-experiment-assurance-lifecycle-v1`.

This is a separate same-owner runtime replay of the frozen v8 packet. It does
not change v8 protocol bytes, thresholds, selected checkpoints, fixtures,
outputs, claim ceiling, or nonclaims. The existing v8 report remains the
scientific reference. No A2A reviewer task is part of this run.

## Frozen target and question

- Protocol: `theseus-replacement-loss-v8-ridge-primary-causal`.
- v8 packet SHA-256:
  `b6bc9563f242bbf20b9acf0decf40798481538930b17be1342cfca47f8158fc2`.
- v8 protocol SHA-256:
  `a1568874241ba4ddd609355857675ebad8e49f8d427a90e9029ce52f4750bfed`.
- Question: do the nine frozen cases and their controls pass when the frozen
  runner reconstructs candidates and evaluates the held-out outputs in a
  clean, same-owner Cloud Run CPU environment?
- Baseline: the exact v8 primary custody packet and the existing separate
  clean-runtime result. No threshold or fixture retuning is allowed.
- Acceptance for this replay: `INTERNAL_PASS` only when the frozen evaluator
  reports `PASS`, all nine cases report `PASS`, there are no case blockers,
  and there are no resource blockers. Any missing case, failure, digest
  mismatch, or timeout is `INCONCLUSIVE` or `FAIL` as recorded by the runner.
- Repeat and stopping rule: one task, one execution, zero retries. A failed or
  timed-out execution is retained and is not retried under this packet.

## Runtime and custody

- CPU only; one Cloud Run Job task; 4 vCPU, 16 GiB memory, 2-hour task timeout,
  one task in parallel, zero retries.
- The input tar headers project 4,145,027,464 uncompressed bytes in the primary
  working set. Cloud Run's default writable filesystem is memory-backed. The
  prior local run measured 3,633,561,600 bytes peak RSS, but that RSS and the
  scratch projection were not measured together in Cloud Run; the packet keeps
  that uncertainty explicit and does not retry a resource failure.
- Deploy only to the user-configured GCP project `secops1219`, region
  `us-east1`; no public service endpoint or schedule.
- The container is built only from source files matching the frozen v8
  manifest plus this separately digest-bound adapter. Its Python base image is
  pinned by the Linux/amd64 manifest digest, the direct Python requirements
  are version-pinned, and the final image digest and complete installed
  dependency inventory are recorded for the exact runtime.
- The exact staged v8 archives and the minimum pinned v7/v6 verification
  archive are mounted from private Cloud Storage buckets. The job identity has
  read-only access to input and write access to output. No credentials are
  included in the image or job environment.
- The run packet binds the exact temporary Artifact Registry repository,
  Cloud Build staging bucket, input and output buckets, dedicated job service
  account, separate least-privilege build service account, and bucket privacy
  and lifecycle settings.
- Model loading is offline. The container makes no model-provider or reviewer
  API calls. All cloud objects remain private and are copied back to the
  external owner-custody SSD before temporary cloud storage is removed.
- The user explicitly authorized this Cloud Run execution. That authorization
  is not a third-party license determination. The exact OPT and GPT-2 model
  revisions do not declare a license in their model metadata; AMD-Llama-135m
  declares Apache-2.0. No legal review is claimed.

## Metrics and assurance boundary

The cloud job records candidate reconstruction, the frozen evaluator outcome,
all nine case statuses, controls, output-array digests, dependency inventory,
container image digest, and runtime identity. It reports `INTERNAL_PASS` only
for the frozen metric acceptance rule above. The v8 replicator additionally
requires the same OS platform for `REPRODUCED_LOCAL`; its recorded baseline is
macOS arm64 and this Cloud Run environment is Linux, so the strict v8
reproducibility field is preserved independently and may remain
`INCONCLUSIVE` even when cloud metric gates pass.

This replay is same-owner evidence. It cannot establish `INDEPENDENT_ACCEPT`,
architecture-general validity, consciousness, alignment, or production
authorization.

## V1 execution record and V2 continuation

The first one-shot execution used run packet
`96c19b8b20efe4f2c579849d235fb8297cdf8d7d5b392cb88ce3de2bcb1cf7fb`, image
`sha256:62b28a38026da48d57fb5310a2bc2962c46c21e26e68af35c2a6c53b4e31b973`,
and execution `theseus-v8replay-1f9f1ac6-8c4tx`. Cloud Run reports task exit code
1. The adapter then failed while opening
`replica/replacement-loss-v3-report.json`; the frozen v3 runner's source has an
early candidate-reconstruction blocker path that writes
`replication-report.json` and returns before creating that evaluation report.
The source path and adapter traceback explain the failure, but the run did not
export its exact blocker list. V1 is retained as `INCONCLUSIVE`; no held-out
evaluation result is claimed.

V1's packet forbids retrying that run identity. One separately frozen V2
operational continuation is allowed by the user's same-owner Cloud Run
authorization. It changes the adapter's result capture so an early-stop
replication report is exported with `INCONCLUSIVE_BEFORE_HELD_OUT_EVALUATION`
instead of crashing. The V2 packet binds the V1 packet digest plus the exact
dispatch, start, pre-execution job, execution, task, and log receipt digests.
The scientific archives, checkpoints, evaluation code, thresholds, fixtures,
claim ceiling, and nonclaims remain the same. V2 is one task with zero retries;
its output is preserved whether it passes, fails, or stops early.

V1 receipts, input archives, packet, and image are retained under owner custody
on the external SSD. V2 has its own input/run packet digests and execution
receipt. A V2 adapter result that captures the frozen runner's blockers is
useful local diagnostic evidence, not a scientific pass unless the frozen
acceptance criteria are met.

V2 execution `theseus-v8replay-1f9f1ac6-m4kcf` exited 1 before model loading.
Its input manifest bound `pinned-v7-verification.tar` at 672,768 bytes with
SHA-256 `bf4aaa44f139c5ee5989c987d486a3707a7c37327c1ee705b0bfa860fdad3936`,
while the private input bucket contained the clean archive at 522,240 bytes
with SHA-256 `ba5ca24ea132158be17d359087b59b3de30ba54230d83534db462feca953b972`.
The three phase archives passed the container's size and SHA-256 checks before
it reached the pinned archive. This is a custody-manifest mismatch, not a
scientific outcome. V2's task, execution, and logs are retained; its packet is
not retried.

One further operational continuation is separately frozen to use the clean
SSD archive whose size and SHA-256 match the private bucket, and to bind all V2
failure receipts. Before dispatch, the archived object is downloaded and
compared byte-for-byte with that source. The v8 experiment packet, phase
archives, model checkpoints, thresholds, fixtures, and claim ceiling stay
unchanged. The continuation has one task and zero retries. It reports a
scientific outcome only if the frozen evaluator actually runs.

## V3 execution result

The third packet is `aa5e177e92470279ede49fd9794a40f020f4c5c883b9f9b62d3fdb7bf217c244`;
its input manifest is
`1a756b8a8a3abb4b98c83533261184e5f6cdb14b738524c57c3fddc7fef05d60` and its
input-object preflight is
`7246d814dc767cf5d18ccf03b6d57e24afdbe436a40386b43c77ba16199d0ef0`. The
one-shot Cloud Run execution `theseus-v8replay-1f9f1ac6-5s7qg` completed
successfully in 8m14.85s using image
`sha256:b7e7c4a2bdfa6658cdee4d1b6d6ec2278ea9b409a390224e77b9ea8323599e83`.
The full output bundle is retained on the external SSD. Its output manifest is
`730c4ac8e248128b0e01c8f4437c9ba71dd210df3341fa82a283c1d5c9ffb344`; all 93
listed payload files passed local size and SHA-256 verification.

The scientific result is `INCONCLUSIVE`, not `INTERNAL_PASS`. Candidate
reconstruction reached all 9 family/seed cases, but adaptation-capture and
candidate-weight digests mismatched in every case (18 blockers total). The
frozen runner therefore stopped before held-out evaluation: no causal,
behavioral, activation-rescue, compositional, adversarial, or control result
was produced in this cloud execution. The separate frozen primary report's
`PASS` is its prior local result; it does not override this cloud replication
failure. The strict v8 report also records candidate reconstruction, source
identity, dependency, and prediction/output digest checks as not reproduced.

`INTERNAL_PASS` applies only to local verification of the downloaded output
manifest and files. Assurance for the scientific reproduction remains
`INCONCLUSIVE`; reproducibility remains `INCONCLUSIVE`; reviewer status is
`EXTERNAL_REVIEW_UNAVAILABLE` (no external review was submitted). The claim
ceiling remains
`LOCAL_INDEPENDENT_AFFINE_PROJECTION_REPLACEMENT_CAUSAL_PREDICTION_OBSERVATION`.
This result does not establish independent science, architecture-general
validity, consciousness, alignment, or production readiness.

No further Cloud Run execution is authorized by these frozen packets. The next
scientific task is a bounded failure analysis of the cross-runtime adaptation
capture and candidate-weight divergence, including the reported source and
dependency identity mismatches. Keep v8's fixtures, thresholds, and claims
frozen; any revised equivalence rule or runtime protocol needs a new packet.

## Cost and cleanup

Cloud Run compute is bounded by one 4-vCPU / 16-GiB task and a 2-hour timeout.
At the currently published Tier 1 rates, the compute ceiling is about $0.75
before any free tier, plus temporary image build, image storage, Cloud Storage,
operations, and logging charges. No long-lived service is created. After local
output verification, the job, three buckets, Artifact Registry repository,
both dedicated service accounts, and locally pulled image copies were removed.
The project-wide Cloud Run, Cloud Build, Artifact Registry, and Storage API
enablement remains. Cloud Logging entries and Cloud Build build-history records
remain under project retention; the exact experiment receipts and logs are
also on the external SSD.

The result packet on the external SSD binds the input manifest, container
source manifest, exact image digest, run configuration, output manifest, and
all output file hashes. It records the actual result separately from the
planned acceptance rule.
