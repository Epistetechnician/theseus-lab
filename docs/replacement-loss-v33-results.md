# Replacement-loss V33 terminal preparation receipt

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v33-relu-geometry-v1`,
`global-experiment-assurance-lifecycle-v1`. Revision:
`202f459d6d89df9a038d78d58e0af825d287fcd8` plus the frozen uncommitted packet.

**Scientific outcome: INCONCLUSIVE.** The global row-disjointness check
stopped preparation on the second case, before any replacement module was
fitted, before challenge generation, and before causal evaluation. Adjacent
case seeds with small partition offsets reused input streams. One guide
state and its preparation inputs were retained; zero replacement states
were written. The terminal custody root was not reused.

Engineering assurance: `INTERNAL_PASS`, 16/16 constructed tests in each of
two isolated runtimes. Those tests did not catch cross-case stream reuse;
scientific execution did. Scientific reproducibility: `NOT_RUN` because
the phase aborted during preparation. Reviewer: `NOT_SUBMITTED`.

| Binding | SHA-256 |
|---|---|
| Initial packet | `3de9ce8a82cf7695a38e8b22fd768a0e1533d6e4f667505e454c68c481108349` |
| Initial freeze file | `fc3769f2329adc4829624fcaf86f493e36068dfe9ea4324313549df6082cb797` |
| External error receipt | `7b61861b0cdc43d852ee49248e6db46ae2a95cf80c2e15b11fab9b73abe056e9` |

Partial data, guide weights, source archive, runtime, and error receipt remain
under `/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v33/development/primary`.
Retained partial artifact bytes: 724,366. Training runtime: Python 3.12.13,
Torch 2.6.0, NumPy 2.4.6, deterministic CPU, one thread, macOS 26.6.1 arm64.
Monetary spend $0; energy and failed-process resource totals were not measured.

The correction is a [new V33R1 packet](replacement-loss-v33r1-relu-geometry.md),
using namespaced streams and an added regression check. It excludes every
V33 partial preparation input and binds this error receipt. This packet
remains unchanged and supplies no learned-replacement finding. The related
[V32 row audit](replacement-loss-v32-row-overlap-audit-v1.md) records the
predecessor's cross-case reuse without changing its frozen verdict.

Claim ceiling: `LOCAL_SYNTHETIC_RELU_GEOMETRY_WEIGHTING_AND_CAUSAL_LADDER_ONLY`.
There is no confirmation, novelty, pretrained-model, architecture-general,
metaphysical-identity, consciousness, alignment, independent-science, or
production claim. The next local gate is qualified fresh-data development
under V33R1; stronger scientific authority remains a separate external gate.
