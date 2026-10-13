# Replacement-loss V19 development results

State slices: `theseus-lab-sandbox-v1`, `theseus-replacement-loss-v19-causal-fit-qualification-v1`, and `global-experiment-assurance-lifecycle-v1`.

- Outcome: `NO_QUALIFIED_DEVELOPMENT_CANDIDATE`; selected lambda: none.
- Assurance: `REPRODUCED_LOCAL`; reviewer: `NOT_SUBMITTED`; blind status: `NOT_BLIND_SAME_OWNER_REPLAY`.
- Packet SHA-256: `7cb5c149f9ad00e137c1bca47299d8dae1c5d992f40adb5d29a582502595bded`; scientific result SHA-256: `86df750504a56fe5f4a34a4b4dbaee5ed5370bccbd45d257a912586ff614d1ed`.
- Repository revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`; the worktree was dirty with 267 entries at execution. The V19 packet digest binds the scoped protocol files.
- Claim ceiling: `LOCAL_CAUSAL_VALUE_PARETO_QUALIFICATION_ON_ONE_REUSED_OPT_CHECKPOINT`.
- Custody: prompts, captures, intervention targets, candidate tensors, and row-level metrics remain in external owner custody.
- Runtime: Python 3.12.13, PyTorch 2.6.0, Transformers 4.57.3, NumPy 2.4.6; deterministic algorithms, one Torch thread. Primary/replay wall times were 341.53/346.71 seconds; peak RSS was 2.42/2.54 GB, within the 4-hour/16-GiB caps. Cost `NOT_APPLICABLE`; energy `NOT_MEASURED`.

## Predeclared global selection screen

A weight had to pass all five checks in every one of nine seed-by-stratum cells: effect-MSE ratio to same-row zero-effect baseline <=0.90, value-MSE ratio to paired pointwise <=1.25, continuation CE increase <=0.05, normalized next-logit error <=0.05, and complete cell coverage.

| Lambda | Equal-cell mean effect ratio | Gates passed | Qualified |
|---:|---:|---:|:---:|
| 0.05 | 0.6717 | 3/5 | False |
| 0.10 | 0.6526 | 3/5 | False |
| 0.20 | 0.6466 | 3/5 | False |
| 0.35 | 0.6523 | 2/5 | False |
| 0.50 | 0.6581 | 3/5 | False |
| 0.75 | 0.6777 | 2/5 | False |
| 1.00 | 0.8493 | 1/5 | False |

The cell-level failures explain why the mean was not enough. No weight reduced
effect MSE by 10% in all nine cells: the best was lambda `0.10`, which passed
5/9 effect cells and had a worst-cell ratio of `0.9142`. No causal weight met
the 1.25 pointwise value-retention bound in any of the nine cells. At lambda
`0.20`, the lowest equal-cell mean effect ratio was `0.6466`, but only 5/9
effect cells passed and the worst value ratio was `2.7038`.

| Lambda | Effect cells passed / worst ratio | Value cells passed / ratio range | CE cells passed / max increase | Next-logit cells passed / max error |
|---:|---:|---:|---:|---:|
| 0.05 | 3/9 / 0.9390 | 0/9 / 1.380–1.622 | 9/9 / 0.0223 | 9/9 / 0.0311 |
| 0.10 | 5/9 / 0.9142 | 0/9 / 1.580–2.130 | 9/9 / 0.0277 | 9/9 / 0.0395 |
| 0.20 | 5/9 / 0.9156 | 0/9 / 1.952–2.704 | 9/9 / 0.0490 | 9/9 / 0.0228 |
| 0.35 | 3/9 / 0.9327 | 0/9 / 3.122–9.090 | 9/9 / 0.0299 | 8/9 / 0.0624 |
| 0.50 | 3/9 / 0.9403 | 0/9 / 3.250–4.827 | 9/9 / 0.0338 | 9/9 / 0.0383 |
| 0.75 | 3/9 / 0.9760 | 0/9 / 9.545–26.226 | 9/9 / 0.0498 | 7/9 / 0.0665 |
| 1.00 | 3/9 / 1.3053 | 0/9 / 227.375–966.976 | 8/9 / 0.1042 | 2/9 / 0.2933 |

The detailed aggregate cell report remains in external custody; SHA-256
`5313eede571590a18e9134c6085aaf60eeed770ae30eb8d820f16437f869e6a7`.

This was development and global candidate selection on one reused OPT/ReLU checkpoint. It did not use or open V18 challenge data. A local replay is not independent acceptance.

No confirmation packet or confirmation fixtures were created, and no challenge entropy was drawn. V18 challenge fixtures were not used by this V19 run. Further training requires a new method and a newly frozen packet; blind external replication and independent acceptance remain separate gates.
