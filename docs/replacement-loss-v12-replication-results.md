# Replacement-loss V12 results: content-addressed V11 replication audit

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v12-replication-normalization-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol: `theseus-replacement-loss-v12-replication-normalization-audit`.

## Verdict

- V12 assurance: `REPRODUCED_LOCAL` for the post-run, content-addressed audit
  of the completed V11 artifacts and `FAIL` outcome.
- V11 scientific outcome: `FAIL`.
- V11 original exact-comparator assurance: `INCONCLUSIVE`.
- Reviewer status: `NOT_SUBMITTED`.
- The V12 primary and replica audit reports are byte-identical. This is
  same-owner, same-host local reproduction, not blind replication or
  independent acceptance.

V12 resolves whether V11's differing candidate-set manifest digests reflected
different candidate content or only runtime-location metadata. Its normalized
candidate, challenge, prediction, and scientific-result identities match.
V11's own formal comparator result remains `INCONCLUSIVE`, and its scientific
result remains `FAIL`; V12 does not upgrade either V11 status.

This was explicitly a post-run audit. The V11 outputs had been observed before
V12 was frozen. V12 did not rerun model inference, generate fixtures or
prompts, fit candidates, or change the V11 packet or results.

## Frozen identity and outcome

- V12 packet digest: `7bd26b9f76105fdfe4e78f0a468bc29b0b89313aea34a40c4e648ae50fe10ac1`.
- V12 freeze-file digest: `b89eac0884bd7022c3e4836263cbd03d4379c5aa59b64f3681994b63d2bffc29`.
- V12 external packet archive digest in both audit roots:
  `6b2dd460fb5412f4c548b73ce3ee19f0e33036e2b9dae9b79bb4766908d296de`.
- Deterministic reconciliation report digest in both audit roots:
  `51aa67fa50281904916722042eb2c8c7e390b0a685c38e207985cef3a8a99221`
  (19,879 bytes).
- V11 packet digest:
  `2bd4e9903b2611ac454a0ff26ffabb051231bd2469f6895aee67504d8c99c582`.
- V11 packet archive digest:
  `8b92f9f762c172c2c6ae49c12447b3906124fc4e6378b9793bd6cd6f0083f08c`.
- Fixed target:
  `grushaprasad/opt125m-rootp-surprisal-curriculum@8af2a2926bc772599ccb3b4e7ad7b40b35259a63`.
- Target selection digest:
  `f5078a88b8094f58fcc1de282d52afe779ae672bc7d8c2ffe6dca1e8941b057f`.
- Target tree digest:
  `6e4c3445311ab3d03830414693cc3c19c6ebfe85176a501629476cf7e2a81466`.

The V12 packet and its archive remain separate from the V11 frozen packet.
V11's raw artifacts remain in its original external custody; the V12 custody
roots contain the frozen V12 packet and one reconciliation report each. The
two roots are `theseus-lab-sandbox-v1-replacement-loss-v12-audit-primary` and
`theseus-lab-sandbox-v1-replacement-loss-v12-audit-replica` under the external
`PrimaryED/ResearchArtifacts` custody root. No model weights were copied into
the repository or V12 packet.

## Reconciliation evidence

The primary and replica reports have the same SHA-256 and are byte-identical.
All six predeclared V11 artifact groups match by relative path, byte count,
and file SHA-256 (64/64 files):

| Artifact group | Files | Shared tree SHA-256 |
| --- | ---: | --- |
| Visible and adaptation fixtures | 4 | `2cba0761341a7d05cbff6d5d42567fdac05986eee7e27291a223f71538826218` |
| Adaptation captures | 6 | `a47f7e8d577f07f52a2927a3331866253aa4d708ddc722e470faa0d2ef71ed76` |
| Candidate states | 12 | `f867d5763dd34e1d1ef7fda07bf390ebe652a91c8cab7e7e1c953cf55ffdc7b3` |
| Challenge entropy | 3 | `5fc10f56e339ed0170e129ebb99e13fa977aa402134cdef59449772e51b7e4b8` |
| Challenge prompts and interventions | 9 | `9c69970a53ad0e8c9fea5a734cd9e08abc83eba0e3d5e1aca00ab4c9f734ed7c` |
| Prediction arrays | 30 | `a83cbd8a1e24621ebe8fcae5c27bffc0cbb5b837ec00fc1fc19065944ef2e3e8` |

The normalized content identities match across primary and replica:

- Candidate content: `eabbffff6af8e4922d14963b1482fde79d7aa97b32f600b872d65660d163b5d4`.
- Challenge content: `9b393a0c798eb3118f3ee8413c118fde531f700d85a742d2fd9ffd1a2016d4ef`.
- Prediction content: `474362cc75bbc3c7dd810441005bb379c3bd7062c4e371992c0d4d0e1fd0eb5c`.
- V11 scientific results: `57fea24a403cbc91cb5a9f392d9336a33d9011fe201dd7469130085e39a86b9b`.
- Prior-failure cell summary:
  `219dd6d145158d4a73f8bec8068e90cd8120e28908409b80102dd328060ac022`.

The original V11 candidate-set digests remain different:
`1a59caee75c119749104a5901b754f8cdb9ed679f3575a154a25ac001372ed27` and
`93db23109a40618cde3680d57b4da9c438e7b4aa145fac0d9bcc9620e5a7aab7`.
V12 records those original values without rewriting them. Its content identity
separates stable model, fixture, candidate, and result content from runtime
paths, and preserves executable path and host provenance in distinct fields.

## Runtime and resource record

Both audits used Python 3.12.13 on macOS ARM64 with the same dependency lock
(`0469ff6948f4866ea4df13d88c53a27adab3141ecb42af1325d31086ca8458d1`) and
the same versions of PyTorch 2.6.0, Transformers 4.57.3, NumPy 2.4.6,
Tokenizers 0.22.2, Safetensors 0.8.0, and Hugging Face Hub 0.36.2. The
executable paths are distinct; the owner and host are the same. These are the
V11 experiment-run budgets recorded by the audit, not additional V12 model
execution:

- V11 primary: 786.05 seconds; peak RSS 3,840,016,384 bytes.
- V11 replica: 774.75 seconds; peak RSS 3,906,912,256 bytes.
- V12 audit: local CPU-only read and hashing; no model inference.
- Monetary cost: `NOT_APPLICABLE`; energy: `NOT_MEASURED`.

## Claim ceiling and next gate

Claim ceiling:
`LOCAL_REPRODUCIBILITY_OF_V11_FAIL_OUTCOME_AND_ARTIFACTS_UNDER_CONTENT_IDENTITY`.

V12 supports only the local conclusion that the already completed V11 artifact
sets and `FAIL` outcome reconcile under the frozen content identity in two
distinct same-owner Python runtimes. It does not establish that the original
V11 inference procedure was rerun by the V12 auditor, blind replication,
independent science, a positive causal-fit result, fresh-checkpoint or
architecture-general validity, novelty priority, general mechanistic
understanding, consciousness, alignment, human equivalence, or production
readiness. No reviewer task was submitted and there is no `INDEPENDENT_ACCEPT`.

The next scientific step is a newly frozen independent-replacement protocol
with intervention predictions bound before held-out outcomes are revealed.
The next assurance gate for that new claim is a separate blind replicator
running its frozen packet. V12 supplies no authority to spend, deploy, publish,
or raise the claim ceiling.
