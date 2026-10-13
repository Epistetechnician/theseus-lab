# Replacement-loss v8 external replication handoff

State slice: `theseus-lab-sandbox-v1`  
Handoff version: `replacement-loss-v8-external-replication-handoff-v1`  
Assessment date: 2026-09-25

## Frozen local evidence

- Protocol: `theseus-replacement-loss-v8-ridge-primary-causal`.
- Frozen v8 packet SHA-256: `b6bc9563f242bbf20b9acf0decf40798481538930b17be1342cfca47f8158fc2`.
- Primary report SHA-256: `317b75f2c448493dd07acba78f1152221bcd6ff96b642de476d2f1d67402aec6`.
- Same-owner, separate-clean-runtime report SHA-256: `8339b4f57ef8e3d24048e457007a1ef0be7c232da77ed2bd558937a9c0e4530d`.
- Local status: `REPRODUCED_LOCAL`; all nine checkpoint-by-seed cases passed.
- No external review task or signed decision has been recorded; see the latest
  live no-task preflight below.

This supports the frozen local claim for one independently fitted affine MLP
output projection in each selected OPT/ReLU, GPT-2/GELU, and Llama/SiLU-gated
checkpoint. It does not establish whole-model identity, all-architecture
validity, independent science, consciousness, alignment, or production
authorization.

## External preflight on 2026-09-25

The configured reviewer public key file matched the registered fingerprint
`ed25519:676cb4e8f55ee166d1b0e21aa3de91a2306f654bea64ba8404e6dfd48699c859`.
The configured Agent Card GET returned HTTP 404 with `ERR_NGROK_3200` (the
ngrok endpoint reported offline). The live card, health, contract negotiation,
and authenticated no-task readiness checks therefore remain incomplete. No
intake request was sent, no task was created, and no artifact bytes were
transferred. The configured profile does not identify an alternate secure
file-transfer channel, so none was guessed.

The selected OPT and GPT-2 checkpoint cards do not declare a license; the AMD
Llama card declares Apache-2.0. Do not transfer the first two checkpoints until
the replicator has an independently verified lawful acquisition path or the
custodian confirms a permitted transfer basis. [OPT checkpoint card](https://huggingface.co/kanishka/opt-babylm2-rewritten-clean-spacy_no-num-adj-earlystop-strictest-bpe_seed-211_1e-3/tree/baa6df5efc0d9f71306d115e64da230ce496b773), [GPT-2 checkpoint card](https://huggingface.co/ai-forever/rugpt3small_based_on_gpt2/tree/a9307e696cd3c5b7f953ff4cb19d76a4d81821d5), [Llama checkpoint card](https://huggingface.co/amd/AMD-Llama-135m/tree/9a224a07b9b0210299d8c19b9e694ac56ed3ba1d).

## Replication packet requirements

The existing v8 `replicate_packet()` path verifies reconstruction, output
array digests, dependency identity, source identity, and use of a distinct
runtime against the primary custody root. That is useful `REPRODUCED_LOCAL`
evidence, but it does not provide a blind external run mode: its comparison
inputs include the primary candidates, predictions, outcomes, and report.
Keep that frozen procedure unchanged.

The external replication continuation must preserve a staged reveal:

1. Freeze a digest-bound handoff manifest for the exact v8 packet, runner,
   dependency manifest, checkpoint revisions and file digests, claim ceiling,
   nonclaims, and retention rules. Preserve the original v8 packet digest.
2. Give the replicator the code, model bytes they may lawfully access, and the
   visible/adaptation inputs. The replicator independently fits and freezes
   the treatment candidate for each of the nine cases plus all five controls
   per case (54 candidate tensors total), then returns a candidate-manifest
   digest before receiving held-out challenge contents.
3. Release the held-out challenge fixtures through an agreed secure file
   channel only after that candidate digest is recorded. The replicator runs
   the frozen behavior, causal, activation-rescue, compositional, adversarial,
   and control checks in a distinct clean runtime across all three families
   and three seeds.
4. The replicator freezes its output arrays, aggregate report, runtime and
   dependency identities, and complete output manifest before receiving any
   primary prediction or outcome files.
5. Only after that receipt is immutable may the custodian release the primary
   reference outputs for byte/digest comparison. Record mismatches,
   inconclusive cases, or missing cases explicitly; do not tune thresholds or
   replace fixtures in the frozen v8 identity.
6. Have the independent reviewer verify the exact final target, packet,
   artifact manifest, claim ceiling, and nonclaims. Keep signed external
   review distinct from the replicator's execution report.

The handoff must include an exact entry manifest for each transferred archive;
all entries, including cache metadata or operating-system sidecars, must be
listed and classified. Prompt text, captures, model assets, candidate files,
predictions, outcomes, and row-level diagnostics remain in owner custody until
the staged transfer is authorized and an agreed secure channel is available.
The A2A message itself must contain only the agreed digest-bound schema fields,
never artifact bytes, raw prompts, traces, credentials, or owner-only paths.

## Next gate

Continue external review only after the responsible operator, custodian, and
registry administrator provide the independently controlled attestations,
trust anchors, and signed target record through their authorized owner paths.
The custodian must also identify an agreed secure artifact-transfer channel
and resolve model-asset transfer rights; the reviewer must verify the exact
bundle bytes through that channel. Then rerun authenticated no-task readiness
for this target and require `ready: true`. Any artifact or target change needs
a new frozen manifest and digest. Submit only after the full preflight passes
and the exact target has explicit user authorization.

The reviewer service is currently reachable, but the exact target remains
blocked by its no-task readiness response below. The current result stays
`REPRODUCED_LOCAL`; local work cannot substitute for a blind external
replication, `INDEPENDENT_ACCEPT`, human-calibrated alignment evidence,
consciousness research, or production authorization.

## Live no-task preflight, 2026-09-27

State slices: `theseus-lab-sandbox-v1` and
`independent-review-lifecycle-orchestration-v1`.

The live reviewer Agent Card and `/healthz` returned HTTP 200. The card
advertised HTTP+JSON 1.0, required bearer authentication, and the lifecycle v1
extension with contract digest
`57ebad516d06293ed1e41bff0fde2cc199f6f7f482a95be200d36a3d99115b2d`, matching
the local canonical contract. The registered reviewer key fingerprint also
matched its pin. An authenticated, read-only pre-task check for target
`87bcaba185c882f8f19201dfe3f8fe42a7e3e856d8ad6f4141d4683f5fb88f61` returned
`ready: false` at `2026-09-27T23:01:15.494Z`.

The readiness response identified these blockers:

- `CUSTODY_ATTESTATION_NOT_PROVIDED` and `CUSTODY_TRUST_ANCHOR_MISSING`
  (`custodian`).
- `PROVENANCE_ATTESTATION_NOT_PROVIDED` and
  `PROVENANCE_TRUST_ANCHOR_MISSING` (`operator`).
- `REGISTRY_ADMIN_TRUST_ANCHOR_MISSING` and `TARGET_RECORD_MISSING`
  (`registry_admin`).
- `VERIFIED_BUNDLE_MISSING` (`operator`); reviewer-side secure import and byte
  verification have not occurred.

The external SSD packet was checked without extracting or displaying its
payloads. Its 14 artifact-manifest entries matched their SHA-256 values; the
three staged archives contained 135, 22, and 217 members, respectively, with
member names, sizes, and hashes matching their member manifests. The artifact
manifest digest is
`06bfab2c91535924f8673de8752fe063690f0c8e95927070505caa4c46180d6f`. The A2A
review-packet digest is
`04064f608bb6a2972c451cd857d4d55f82a937d3bcfb9ee273d687bc23d1392f`; the v8
experiment packet digest remains
`b6bc9563f242bbf20b9acf0decf40798481538930b17be1342cfca47f8158fc2`, and the
protocol digest is
`a1568874241ba4ddd609355857675ebad8e49f8d427a90e9029ce52f4750bfed`. The top-level
`README.md`, `target.json`, and `target-digest.txt` are operator-side staging
metadata outside the 14-entry payload manifest: the README gives local
handoff instructions, `target.json` is the canonical local target input, and
`target-digest.txt` is its checksum convenience copy. They are excluded from
payload transfer. If any is later transferred as an artifact, freeze a new
manifest and target digest. The frozen packet was not changed.

No intake request or review task was created, and no artifact bytes were
transferred. There is no signed reviewer decision. Continue to report the
scientific result as `REPRODUCED_LOCAL`; the external acceptance gate remains
closed until the listed owner attestations, trust anchors, target record, and
verified secure import are provided.

## No-task preflight refresh, 2026-09-27 23:18 UTC

State slices: `theseus-lab-sandbox-v1` and
`independent-review-lifecycle-orchestration-v1`.

The authenticated pre-task check was repeated at `2026-09-27T23:18:45.388Z`.
It returned `BLOCKED` / `ready: false` with the same seven blockers above,
binding packet `04064f608bb6a2972c451cd857d4d55f82a937d3bcfb9ee273d687bc23d1392f`,
protocol `a1568874241ba4ddd609355857675ebad8e49f8d427a90e9029ce52f4750bfed`,
and target `87bcaba185c882f8f19201dfe3f8fe42a7e3e856d8ad6f4141d4683f5fb88f61`.
The configured local review-lifecycle directory contains no filename for this
Theseus target and no custody-attestation, provenance-attestation,
role-specific trust-anchor, or secure-import record. Its available transfer
receipt is explicitly digest-check-only and says no authorized secure channel
is configured. This confirms that there is no local record or documented route
to apply to this target; it does not establish that owner-held records do not
exist elsewhere.

No request, task, signature, trust anchor, target record, or transfer was
created. The next actionable inputs must come from the named custodian,
operator, and registry administrator through their authorized provisioning
paths; the custodian must also identify the agreed secure import channel and
resolve the model-asset transfer basis. Once those inputs exist, rerun the
read-only preflight and preserve the frozen target. Do not submit unless it
returns `ready: true` and the exact target is explicitly authorized.
