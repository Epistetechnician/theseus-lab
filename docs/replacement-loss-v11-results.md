# Replacement-loss V11 results: causal-fit remediation on reused OPT

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v11-known-opt-causal-fit-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol: `theseus-replacement-loss-v11-known-opt-causal-fit-remediation`.

## Verdict

- Scientific outcome: `FAIL`.
- Primary execution assurance: `INTERNAL_PASS`.
- Frozen exact-replication assurance: `INCONCLUSIVE`.
- Reviewer status: `NOT_SUBMITTED`.
- Reproducibility: the primary and replica produced identical experiment files
  and scientific-result digest, but V11's frozen comparator did not accept the
  run because its manifest identity includes the runtime executable path.

The primary comparison was causal-fit minus activation-ridge held-out
next-logit intervention-effect cosine. Its mean paired difference was
`-0.253148`; the clustered 95% lower bound was `-0.564703`, below the frozen
`> 0.05` requirement. The per-seed differences were `-0.271650`, `-0.564703`,
and `+0.076910`. Causal-fit retention failed. Activation rescue and the
zero-intervention control passed. The V9 failure-cell summary was descriptive:
causal-fit passed 1/9 previously missed cells, while activation-ridge passed
2/9. This is not a positive remediation result.

The scientific outcome is the same in the primary and replica. Both have
scientific-results digest
`57fea24a403cbc91cb5a9f392d9336a33d9011fe201dd7469130085e39a86b9b` and prior
failure-cell summary digest
`219dd6d145158d4a73f8bec8068e90cd8120e28908409b80102dd328060ac022`.

## Frozen identity and target

- Repository base revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`.
- V11 packet digest: `2bd4e9903b2611ac454a0ff26ffabb051231bd2469f6895aee67504d8c99c582`.
- Freeze-file digest: `f3a16e77f778c5c82d40e4c5234440b1b614abc746a06243042cefa253f41d41`.
- External packet archive digest:
  `8b92f9f762c172c2c6ae49c12447b3906124fc4e6378b9793bd6cd6f0083f08c`.
- Reused target:
  `grushaprasad/opt125m-rootp-surprisal-curriculum@8af2a2926bc772599ccb3b4e7ad7b40b35259a63`.
- Source V9 checkpoint-manifest file digest:
  `b52bd7b94eb1e14c0453ba9318fcec2b39f0a210ca4c35f88b070bd57cd8e277`.
- Target tree digest:
  `6e4c3445311ab3d03830414693cc3c19c6ebfe85176a501629476cf7e2a81466`.
- Raw model-weight-file digest:
  `aadbd43968ed14753343d18042654e812970320b7194982b59f6d4aaa40c3132`.
- Target-selection receipt digest:
  `f5078a88b8094f58fcc1de282d52afe779ae672bc7d8c2ffe6dca1e8941b057f`.

The exact previously studied V9 OPT/ReLU checkpoint was intentionally reused.
V11 used new synthetic fixture namespaces and three new seeds, with layers 0
and 7 fixed from the prior failure analysis. It does not test fresh-checkpoint
generalization.

## Replication and comparator finding

The primary ran in Python 3.12.13, PyTorch 2.6.0, Transformers 4.57.3, and
NumPy 2.4.6 on macOS ARM64. The replica used the same versions and dependency
lock in a separate Python installation and virtual environment on the same
host and under the same owner. Both runs used one PyTorch thread and
deterministic algorithms.

Byte-level comparison found identical files across all six frozen experiment
artifact groups: 4 fixture files, 6 adaptation captures, 12 candidate states,
3 challenge-entropy files, 9 challenge/intervention files, and 30 prediction
arrays (64/64 files total). The candidate rows, checkpoint identity, target
selection, scientific summary, and prior-failure summary also match.
The hash-only comparison receipt was retained in both external custody roots;
its SHA-256 is
`9feb104de5a4fce251f2a6bb1c873c06c48251cdf8aacee62d8f9b3304f6cb88`.

The formal V11 comparator nevertheless returns `INCONCLUSIVE`. The candidate
set digest hashes the complete runtime record, including the absolute
`python_executable` path. The required separate runtime therefore changes the
candidate-set digest. Challenge and prediction manifest digests include that
candidate-set digest, so they differ as well, even though their underlying
fixture, intervention, and prediction files match byte-for-byte. Using the
same executable path would make that digest match but would fail V11's
separate-runtime check. This is a frozen packet design defect; V11's protocol,
freeze, and results have not been edited to work around it.

Primary run: 786.0 seconds, peak RSS 3,840,016,384 bytes. Replica: 774.8
seconds, peak RSS 3,906,912,256 bytes. Both completed all three phases within
the four-hour and 16-GiB limits and produced no stop receipt. Software
qualification before freeze passed 6/6 V11 tests and 9/9 V10 engine tests in
each existing pinned environment. The separate replica runtime passed the
frozen version, dependency, lock, and packet checks before its full run.

## Custody, claim ceiling, and limitations

The checkpoint remains in its original external V9 custody. V11 stored only a
digest-bound target receipt; it did not copy the model weights. Prompts,
fixtures, captures, candidates, entropy, predictions, interventions, and
row-level outcomes remain in external owner custody. Only aggregate results
and digests are recorded here. No paid compute, cloud worker, GMan node,
provider call, or public write was used. Monetary cost is `NOT_APPLICABLE`;
energy is `NOT_MEASURED`.

Frozen claim ceiling:
`LOCAL_CAUSAL_FIT_REMEDIATION_ON_REUSED_V9_OPT_CHECKPOINT_WITH_NEW_SYNTHETIC_FIXTURES`.
Since the scientific outcome is `FAIL`, V11 supports no positive claim that
causal-fit improved remediation. The result is limited to this implementation,
checkpoint, protocol, and synthetic fixture set. It does not establish
novelty priority, architecture-general validity, general mechanistic
understanding, consciousness, alignment, human equivalence, independent
science, or production readiness.

## Next step

Keep V11 immutable. A new, separately frozen replication packet should hash
candidate content independently of runtime-location metadata while recording
runtime provenance in its own field. Its comparison should bind stable
candidate, fixture, prediction, and scientific-result digests, then repeat
the frozen procedure in a clean second runtime. That packet can resolve the
local reproducibility status; it cannot turn this scientific `FAIL` into a
pass. A later scientific protocol should investigate why adaptation-only
causal-fit underperformed activation-ridge before expanding the claim to a
fresh checkpoint or another architecture family.
