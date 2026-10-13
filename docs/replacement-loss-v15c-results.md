# Replacement-loss V15c results: capacity versus optimization

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v15c-capacity-optimization-diagnostic-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Disposition

- **Assurance:** `REPRODUCED_LOCAL`.
- **Diagnostic result:** `NO_CLEAR_SIGNAL`.
- **Reviewer:** `NOT_SUBMITTED`.
- **Blind status:** `NOT_BLIND_SAME_OWNER_REPLAY`.
- **Claim ceiling:**
  `LOCAL_ADAPTATION_DIAGNOSTIC_OF_RANK_AND_OPTIMIZATION_FOR_ONE_REUSED_OPT_CHECKPOINT`.

The exact frozen V15c procedure completed in the primary runtime and a clean,
separately provisioned Python runtime. The protocol verifier passed, and the
replication comparator found identical bytes for adaptation data, candidates,
fixtures, splits, candidate manifest, and diagnostic results. This is
same-owner, same-host reproducibility; it is not independent review.

## Frozen identity and target

- Protocol: `theseus-replacement-loss-v15c-capacity-optimization-diagnostic`.
- Packet SHA-256: `4df6b7e71505d0a5d69570d6967bb55423a734a3d37d218424aab1dbcb80f291`.
- Freeze SHA-256: `f8afd5dd7ab5c2aa79fd35d5ef41a40d4ca26a74f580ab120af69865d041e8ae`.
- External archive SHA-256, primary and replica:
  `6395c0b251d0d9764d2001ec1a3f9daee4d3be16c6b1c25b58252b47c46d56ed`.
- Source identity: `202f459d6d89df9a038d78d58e0af825d287fcd8+v15c-frozen-working-tree`;
  the executable packet bytes are bound by the packet digest.
- Target: `grushaprasad/opt125m-rootp-surprisal-curriculum` at
  `8af2a2926bc772599ccb3b4e7ad7b40b35259a63`.
- Target selection SHA-256:
  `020aa9b9f74d897d38d960807c4db91af5fe4fb17a1da1fd61afb4a02c7f970f`.

This is one reused, outcome-aware OPT/ReLU checkpoint and fixed layers 0 and 7.
The three seeds and six seed/layer cells are descriptive nested cells, not IID
observations. The experiment evaluates adaptation fit/validation only; it does
not evaluate held-out causal challenges, continuation behavior, replacement
retention, or learned activation rescue.

## Diagnostic findings

The predeclared signal required at least a 10% relative change and at least
five positive cells out of six.

| Comparison | Mean validation relative reduction | Positive cells | Signal |
| --- | ---: | ---: | --- |
| Rank 512, update 200 to 800 | 5.77% | 6/6 | No; below 10% |
| Rank 768, update 200 to 800 | -1.58% | 3/6 | No |
| Rank 768 vs 512 at update 200 | 1.17% | 4/6 | No |
| Rank 768 vs 512 at update 800 | -6.28% | 2/6 | No |

There were no predeclared overfit-signal cells. The result says this run did not
meet its thresholds for an optimization or rank-capacity signal. It does not
show that rank or optimization never matters, and it is not evidence that a
replacement preserves the model's behavior or causal mechanisms.

Diagnostic-results SHA-256, identical in both runs:
`109b5b94ff1bbc5e2e2b6cc85de1776e8afafce46048c8268af164791d9f9d5f`.

## Reproduction evidence

Both environments used Python 3.12.13, Torch 2.6.0, Transformers 4.57.3, the
same dependency-lock digest, deterministic algorithms, one Torch thread, and
the same macOS arm64 host. The Python executables were separate. All compared
artifact groups were byte-identical:

| Group | Files | Tree SHA-256 |
| --- | ---: | --- |
| Adaptation captures | 6 | `39ca91ba7f2a000de981b5a445b8c71d9a8e0207b35bdc99fdc1d487f6f7a55d` |
| Candidate tensors | 24 | `67631465c0d9ef6c5ede17e26701930277c231e54eab11c483e58f20b57bfbfb` |
| Diagnostic results | 1 | `0a9866c1708c38d2d9b3546f0eac71e6f048dd41ccfb33687bbd40fb017051e2` |
| Fixtures | 4 | `d7b937753879f38397f1947ec4a2531faecb269862389a856d0a5eba8ab3f4ee` |
| Candidate manifest | 1 | `f7f21511374621ede3e8312ba11fd580e2ff64a7cae9951a8e150d299a4e7fc3` |
| Splits | 3 | `b12555ee1769410406740f19449d704bc6dc7cc9bccb82974a83751b401e64cc` |

| Run | Elapsed | Peak RSS |
| --- | ---: | ---: |
| Primary | 4,824.545 s | 2,601,596,976 bytes |
| Clean-runtime replica | 4,416.867 s | 3,514,056,704 bytes |

The replication report SHA-256 is
`59ff0fffac3eb8b30c910a358dfbd936aac35f5a170c37303abaffd283c253e3`.
The primary and replica runner report SHA-256 values are respectively
`ee78c7855dc154b1ff5a52133d0bc6e37724400799b7518ba6415e4699cacbcc` and
`274eed7c7c3cf2b60b1166340ec00b9ee1a34960dfd3307f4d790247a67befa7`.

## Custody, limits, and next gate

Synthetic prompts, captures, candidate tensors, split identifiers, and
per-cell metric rows remain under external owner custody at
`/Volumes/PrimaryED/ResearchArtifacts/`. They are not copied into this
repository. Monetary cost is `NOT_APPLICABLE`; energy is `NOT_MEASURED`; no
paid or remote compute was used.

The next scientific step is a new frozen protocol for an independently
constructed replacement trained on disjoint adaptation data, followed by
pre-registered predictions on unseen causal interventions, activation rescue,
behavior, compositional prompts, adversarial interventions, and controls. The
candidate must be frozen before those outcomes are revealed. A blind external
replication and authorized reviewer are still required for independent
scientific acceptance. This diagnostic establishes no architecture-general,
alignment, consciousness, or production claim.
