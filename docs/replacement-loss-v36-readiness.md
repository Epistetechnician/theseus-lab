# V36 readiness: frozen source, pretrained execution blocked

State slices: `theseus-lab-sandbox-v1`,
`theseus-pretrained-whole-block-transfer-v36`,
`global-experiment-assurance-lifecycle-v1`. Date: 2026-10-05.
Revision: `202f459d6d89df9a038d78d58e0af825d287fcd8` plus digest-bound
uncommitted source. [Protocol](replacement-loss-v36-pretrained-transfer.md),
[source freeze](replacement-loss-v36-freeze.json),
[scalar receipt](replacement-loss-v36-readiness.json).

## Result

**`INTERNAL_PASS` for engineering qualification. The pretrained study is
`NOT_RUN_CUSTODY_UNAVAILABLE`; no scientific result is claimed.**

The source packet is frozen before selecting checkpoints. Both approved drives
are absent at the final availability check. No V36 intake, fresh checkpoint
selection, pretrained training/evaluation, or scientific replica opened.

| Obligation | Current status |
| --- | --- |
| Full MLP capture, learned replacement, prediction seal and evaluator | Implemented |
| Source, thresholds, fixture rules, selection rules and exact dependency lock | Frozen |
| Engineering qualification | 21/21 tests in each of two isolated environments |
| Current V35 artifact custody verification and historical intake | Blocked: both drives unavailable |
| Fresh OPT/GPT-2/Llama checkpoint availability and rights | Not screened |
| Nine pretrained cases, exact second-environment reproduction and closeout | Not run |
| Independent reviewer | `NOT_SUBMITTED` |

## Exact bindings

- Source packet SHA-256: `370a074309eeea6b57cf5b529892932d5d93474a9381b7ffd47bc1a6bb367458`.
- Source freeze file SHA-256: `c6a8b44ed472e4c5fa82effad5ea514865242eb280d62910ffe79ab66de32739`.
- Dependency lock SHA-256: `d8ca46482cc3e43e61a1994ec0fd49584389267063720d4534da55c55a1b99b7`.
- Identical installed dependency-file digest in both qualification environments:
  `5848cc0ba94e32acae3c4adfe8602a0d7e771fb0c2cc3c7f09e09818e12c1488` (18,740 files each).
- Parent V35 packet: `2b5a23c1a7c1fd5979b384f559ae152cd6f4febd8c5f2d455db05a4abd569656`.
  Its aggregate report/freeze are bound source members. Actual external parent
  bytes could not be reverified on this date.

## What was implemented and qualified

The runner replaces complete MLPs in OPT/ReLU, GPT-2/GELU and Llama/SiLU-gated
decoders with independently initialized nonlinear modules. It fits endpoint,
finite-effect and capped directional-derivative losses using adaptation data
only. Both stage snapshots and all predictions freeze before oracle assessment.

The frozen design has one checkpoint per family and three replacement/corpus
seeds, for nine cases. Each case tests behavior, individual/joint native input
effects, actual activation restoration, restoration prediction, compositional
and adversarial fixtures. The endpoint comparator and random, zero and
swapped-module controls are retained. The positive no-op control actually
reruns the decoder with the original MLPs reinstalled.

A separate numerical closeout verifies prediction/oracle/rescue/weight/capture
bindings, recomputes every native cell and control sensitivity, and checks the
method/aggregate verdict and all reconciliation keys. It does not independently
reimplement decoder capture semantics or confer independent authority.

The 21 tests include all three actual architecture adapters and independent
autograd comparisons; parameter removal/restoration; intervention hook cleanup;
loss cap and repeatability; fixture collisions; checkpoint provenance and
manifest controls; source/custody/capacity failures; terminal preservation; and
a miniature full packet lifecycle. That miniature uses random tiny models,
two updates per module, one seed per family, two rows per partition and mocked
study custody/runtime qualification. Separate closeout recomputes 48 native
cells and 24 gradient readouts for its primary packet. Deliberate prediction
seal, control-verdict, aggregate-verdict and reconciliation omissions are rejected.
It supplies engineering evidence only.

| Qualification environment | Tests | Suite time | Peak process RSS |
| --- | --- | --- | --- |
| Primary | 21/21 | 3.078 s | 335904768 bytes |
| Replica | 21/21 | 3.082 s | 336232448 bytes |

Both use Python 3.12.13 and the exact 34-package lock on macOS arm64, with
distinct prefixes/site-packages and the same base interpreter. Qualification
copied the installed dependency closure offline and verified every file.
Suite times exclude runtime creation; receipt wall times include harness
imports. Tests compare each miniature packet's primary and replica; matching
suite passes across environments do not establish identical random miniature
fixtures or exact pretrained reproduction.

## Failures and corrections retained

1. The initial crash-test fixture omitted its manifest file and errored. The
   fixture was corrected before freeze.
2. Offline uv setup hit the unavailable external cache symlink twice. Blind
   retries stopped. Official cache-directory, no-cache and stdlib-venv options
   were considered; verified offline package copies were used. The existing
   cache and repository runtime were preserved.
3. The initial copied managed Python interpreter could not resolve libpython
   on macOS; the observed primary exit was 134. A replica attempt ran but its
   exit was not retained. Only the owned temporary interpreter links were
   rebuilt with [documented venv symlink behavior](https://docs.python.org/3.12/library/venv.html)
   before freeze.

These are engineering/setup failures, not held-out scientific failures. No
scientific terminal root was repaired or reused.

## Resources, custody, and claim ceiling

The monetary cap and measured paid execution are $0. Local energy and machine
amortization are unmeasured. Each actual runtime is limited to 24 hours, 6 GiB
peak RSS and 24 GiB artifacts, with a pre-fit free-space check of 25 GiB.
Vocabulary is capped at 65,536. The cap was sized before freeze to retain full
logit predictions, controls and rescue arrays; the maximum declared numeric
payload alone is about 15.28 GiB across nine cases. Limits are cooperative,
and full-run time/memory/storage behavior is unknown.

Study artifact custody:
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-pretrained-whole-block-transfer-v36`.
Study runtime custody: `/Volumes/Backup ED/ResearchRuntimes/theseus-lab-v36`.
Instantiated prompts/tokens, assets, captures, weights, predictions, entropy and
oracle outcomes must remain there. No real study assets were generated here.
Only source, rules, tests, dependency pins and scalar/digest evidence are in
the repository. Synthetic test directories were temporary; qualification
runtime copies were removed after final report validation. The existing
repository runtime and cache were preserved. Historical exclusions bind known
registry and retained predecessor JSON records; they do not establish absence
from unretained or non-JSON-only predecessor corpora.

Claim ceiling: `LOCAL_PRETRAINED_TWO_MLP_INTERFACE_REPLACEMENT_ONLY`.
Current evidence reaches engineering readiness only. Pretrained retention,
mapping equivalence, universal identity, novelty, consciousness, human alignment,
independent acceptance and production remain unestablished.

## Exact next steps

1. Reconnect both approved drives. Verify current V35 artifact/result/
   reconciliation/closeout digests, then create the one V36 historical intake.
2. Screen and bind one fresh eligible checkpoint per family after this source
   freeze. Availability is unknown; an ineligible cohort does not justify
   relaxing the frozen contract. No download/acquisition adapter is opened.
3. Prepare two isolated runtimes in approved external custody using the exact
   lock; record interpreter/dependency-file hashes and distinct environment paths.
4. Execute the primary once, the exact replica once, and separate closeout using
   the [frozen commands](replacement-loss-v36-pretrained-transfer.md).
   Preserve any target miss, scientific `FAIL`, resource fault or
   `INCONCLUSIVE` outcome literally.
5. Report `REPRODUCED_LOCAL` only if the actual frozen procedure reproduces.
   Independent science requires the separately authorized reviewer gate.
   Human calibration, consciousness research and production each retain their
   separate proof and authority obligations.

No cloud run, remote worker, review task, publication, commit or merge occurred.
