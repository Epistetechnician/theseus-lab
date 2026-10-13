# Replacement-loss v9 external review readiness

State slices: `theseus-lab-sandbox-v1` and
`independent-review-lifecycle-orchestration-v1`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

Status as of 2026-10-07: **`EXTERNAL_REVIEW_UNAVAILABLE` / `NOT_READY` /
`NOT_SUBMITTED`**. The configured reviewer returned HTTP 404 for both its
registered Agent Card and health route. No V9 A2A review task was created, no
archive was transferred, and no raw prompts, traces, model assets, or outputs
were sent.

Target-specific no-task readiness: `NOT_RUN`; the archive-verification and
independent-authority prerequisites were not met.

## Exact study target

- Protocol: `replacement-loss-v9`.
- Protocol packet SHA-256:
  `9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959`.
- Result: `FAIL`, reproduced locally as `REPRODUCED_LOCAL`.
- Replication manifest file SHA-256:
  `4f759a1396147bf723b9cec5c8f97716128cf02673808a7212da8de41634a61c`.
- Result-binding receipt digest:
  `9c89ac6db660dc7948002279278db7059f3b8fc3d2abf7b15110cc32e439542e`.
- Claim ceiling:
  `LOCAL_CAUSAL_EFFECT_PREDICTION_ADVANTAGE_OF_AFFINE_REPLACEMENT_OVER_CKA_AND_LOGIT_DISTILLATION_THROUGH_THREE_REPLACEMENTS`.
- Reviewer status: `NOT_SUBMITTED`.

The user authorized proceeding with external replication of this exact v9
result. The remaining gates below prevent a safe, independently attributable
submission at this time.

## Current read-only live checks

Checked 2026-10-07 against the exact reviewer host and routes in the registered
profile:

| Check | Result |
| --- | --- |
| Registered reviewer Agent Card | HTTP 404; the registered route is unavailable |
| Reviewer health | HTTP 404; the registered route is unavailable |
| Registered reviewer public-key pin | Local key fingerprint still matches the registered Ed25519 pin |
| Local relay Agent Card and health | Not rechecked; this profile does not record the exact relay endpoint, so no endpoint was guessed |
| V9 task record | No V9 task ID was available to poll; no new task was submitted |
| Target-specific no-task readiness | `NOT_RUN`; archive, transfer, and authority prerequisites remain unmet |

No alternative host or route was tried. The 2026-09-27 checks below are
historical and do not establish current service readiness. The exact V9
protocol packet remains `9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959`;
the local source-member set still matches the frozen worktree packet recorded
in the V9 results. These local checks do not establish reviewer authority,
archive receipt, or `INDEPENDENT_ACCEPT`.

## Historical read-only live checks

Checked 2026-09-27:

| Check | Result |
| --- | --- |
| Registered reviewer Agent Card | HTTP 200; name and `/message:send` URL match the registered profile; HTTP+JSON 1.0 and bearer authentication advertised |
| Reviewer health | HTTP 200; `status: ok` |
| Reviewer lifecycle extension | Advertises `urn:epistetechnician:independent-review-lifecycle:v1` and contract digest `57ebad516d06293ed1e41bff0fde2cc199f6f7f482a95be200d36a3d99115b2d` |
| Local relay Agent Card and health | HTTP 200; same extension digest; `RELAY_ONLINE`, `lifecycle_supported: true`, `task_submission_enabled: false` |
| Registered reviewer public-key pin | Local raw-key fingerprint matches the registered Ed25519 pin |
| Local contract digest | Recomputed digest matches the reviewer and relay cards |

At that time, these checks established transport and contract advertisement.
They did not establish that the reviewer had independent human or
organizational authority, that the archive was received, or that a
target-specific readiness decision was `ready: true`.

## Reviewer coordination request

Use this only in an already authorized reviewer-side coordination channel. Do
not send it to `/message:send`, which is the task-intake route.

> We authorize a blind external replication attempt for the frozen
> `replacement-loss-v9` target. Protocol packet SHA-256:
> `9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959`.
> Result-binding packet SHA-256:
> `edf615792c8e193f74e8b840faeb0e9c2c021eba938c4b20a0d31db1fce55753`.
> Claim ceiling:
> `LOCAL_CAUSAL_EFFECT_PREDICTION_ADVANTAGE_OF_AFFINE_REPLACEMENT_OVER_CKA_AND_LOGIT_DISTILLATION_THROUGH_THREE_REPLACEMENTS`.
>
> Before any task submission, please provide: (1) the exact signed-decision
> field set and canonicalization, including confirmation that the signature
> binds the complete canonical `target_digest`, state slice, review scope, and
> nonclaims; (2) the reviewer-approved secure file-transfer channel, endpoint,
> and archive layout; (3) the identities and public-key fingerprints for the
> independently controlled provenance signer, custodian, and registry
> administrator, plus the exact attestations and schemas each role must issue;
> and (4) the archive-digest verification and authenticated no-task readiness
> procedure for this target.
>
> The blind handoff must keep held-out prompts, intervention fixtures, source
> outcomes, candidate states, and predictions in their required stages. The
> replicator must seal its own candidate states and predictions before the
> custodian releases held-out fixtures or outcomes. Please confirm the
> reviewer can support that sequence and return reviewer-local archive and
> target digest verification. No review task should be created until the
> transfer, authority, and readiness gates pass.

## Remaining blockers

1. **Independent authority is unverified.** For this target, no separately
   controlled provenance signer, custodian, or registry administrator and
   their trust anchors have been identified. The live host and key checks do
   not prove independent control of those roles.
2. **No agreed secure archive channel or layout is recorded for this target.**
   The A2A contract is digest-only; it must not carry the 6.5 GB custody
   payload, prompts, candidate files, predictions, or outcomes. The current
   relay exposes A2A routes, not an authorized artifact-upload interface.
   Blind replication also needs a staged handoff that withholds held-out
   outcomes until the replicator seals predictions.
3. **The signed decision binding needs written peer confirmation.** The live
   card names the v1 decision schema, but the shared contract does not
   enumerate its signed fields or canonicalization. The operator implementation
   requires a signature bound to the complete canonical `target_digest`; the
   reviewer has not confirmed those exact signed bytes for this target.
4. **No target-specific archive or artifact manifest has been frozen or
   independently verified.** Accordingly, the authenticated no-task readiness
   route was not called. Health and card responses are not readiness.

## Safe continuation sequence

1. Obtain reviewer confirmation of the target-bound decision fields and
   canonicalization, plus the exact secure transfer channel and staged archive
   layout.
2. Identify the independently controlled signer, custodian, and registry
   administrator for this packet and verify their trust anchors and records.
3. Freeze the exact staged archive and artifact manifest; verify every file
   digest and classify every archive entry.
4. Transfer the archive through the agreed secure channel and obtain the
   reviewer's archive-digest verification.
5. Run the authenticated no-task readiness check against the frozen target.
   Submit once only if every gate passes; then poll and verify the signed
   decision against the pinned reviewer key and exact target digest.

Until those prerequisites are satisfied, the external status remains
`NOT_READY` / `NOT_SUBMITTED`. The local result remains `FAIL` /
`REPRODUCED_LOCAL`; no experiment bytes or claim ceiling were changed.
