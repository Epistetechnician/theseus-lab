# V39 all-token replacement development ladder

2026-10-06, revision `202f459d6d89df9a038d78d58e0af825d287fcd8` plus
digest-bound source. State slices `theseus-lab-sandbox-v1`,
`theseus-reused-gpt2-all-token-ladder-v39`,
`global-experiment-assurance-lifecycle-v1`.

## Hypothesis and fixed experiment

V38 remains a reproduced FAIL. Its separately frozen adaptation-only audit
found prefix errors substantially above final-token errors. New hypothesis:
supervising the full token interface may preserve sequence behavior and
unseen native intervention effects through sequential learned replacements.
The audit motivates this experiment; it does not prove the cause of every
V38 failure. V39 is a new development packet, not a retry or confirmation
on old held-out cases. V39 outcomes cannot tune this packet.

Reuse precisely `mrm8488/spanish-gpt2`, revision
`fc396ce08c65ea334dcf541de7dc05cac89a29b2`, MLP sites 0 and 2. Recheck both
visible removal thresholds (.10 logit nRMSE, .20 top-1 change). One miss is
NO_QUALIFIED_TARGET with no fitting. Three seeds 36101-36103, JVP and
endpoint arms, two sequential stages, 128 adaptation/donor prompts per seed,
1,024 updates, batch 64, four derivative probes, fresh independent student
initialization, AdamW learning rate .001, weight decay .0001, gradient cap,
and every V36 numerical/control gate remain unchanged. No original target
parameter may remain installed. The endpoint arm is a paired descriptive
comparison; this is not a CKA/prior-art novelty test.

Material change: capture each MLP input at every token in both adaptation
and donor prompts, in the actual earlier replacement context. Pool every
adaptation token exactly once, with equal per-token loss sampling. Pair its
donor at `floor(i*(Ld-1)/(La-1))` within the paired donor prompt. Both
first/last endpoints are retained; donor positions may repeat when La>Ld.
Output mean/variance are computed over the pooled adaptation targets.
This changes training coverage and its normalization statistics together;
the experiment does not isolate those two contributions or provide a
matched statistical effect estimate against V38's different fixtures.

The fitting adapter marks only live adaptation/donor token lists. Marked
captures are padded to 96 solely for stacking, using a validated metadata
mask. The mask/padding never enters teacher/student feature vectors.
Recorded captures store only 768-wide native pooled x, paired donor, all
original donor tokens, prompt lengths and schema version. Separate closeout
checks token counts against source IDs, finite values, feature width and
the exact donor mapping. Native evaluation delegates to frozen V36 code
without changing last-token interventions or sequence NLL semantics.

## Freeze, evaluation and provenance

Freeze code, fixture rules, controls, thresholds and this protocol before
V39 data generation or fitting. V39 protocol domain-separates the existing
fixture generator's preparation streams. Prior V38 visible/adaptation/
held-out whole-text/token hashes are exclusion-only; none are reused as
training or assessment examples. Global text/token disjointness remains
strict, with 2-96 tokens, no truncation, retries or regeneration.

Freeze all 12 candidates before drawing held-out OS entropy. Four strata,
32 recipient/donor pairs per seed/stratum: ordinary, input composition,
interchange composition, adversarial. Seal every candidate/control prediction
before guide-oracle evaluation. Check held-out logit nRMSE <=.10, top-1
agreement >=.95, mean NLL increase <=.03, activation nRMSE <=.15,
actual rescue and rescue-prediction nRMSE <=.10. Single/joint effects require
resolved fraction >=.90, median relative error <=.20, p90 <=.50, median
cosine >=.90. Real no-op controls must pass; random/wrong-module/zero joint
errors must exceed treatment by >=.10 in all strata. PASS requires all
three JVP cases, both stages, every stratum and control. Retain every miss.

Replica rebuilds candidates in the clean second runtime before reading
primary entropy and matches sealed predictions before running its oracle.
Separate persisted-array formulas recheck all outcomes and artifacts.
Both processes share the owner, physical Mac and system libraries; shared
semantics can reproduce defects. No external reviewer is submitted.

## Custody, limits and completion

All model assets, prompts, activations, candidates, predictions and row-level
outputs remain on PrimaryED. Two existing isolated V37R1 prefixes on Backup
ED have exact 34-package lock/file fingerprints. Explicit guide copy stays
inside this project's external custody, verifying all nine pinned files.
No network, provider calls, paid/remote compute, gman node, publication or
deployment. Budget per role: eight hours, 6 GiB peak RSS, 24 GiB artifacts,
25 GiB free-space reserve, deterministic one-thread CPU, paid $0. Cooperative
guards, not OS quotas; full economic cost and energy remain unmeasured.
Changed custody/source, nonfinite data, cap exhaustion, execution error or
reproduction mismatch is INCONCLUSIVE. No same-root retry or retuning.

End state: primary, replica, numeric/token-capture closeout and literal scalar
report. Engineering INTERNAL_PASS; exact reproduction REPRODUCED_LOCAL;
reviewer NOT_SUBMITTED. Claim ceiling
`LOCAL_REUSED_GPT2_EARLY_ALL_TOKEN_REPLACEMENT_DEVELOPMENT_ONLY`.
One reused Spanish-trained guide and English procedural prompts do not
establish semantic correctness, fresh architecture-held-out transfer,
three-family nine-case coverage, whole-depth or mapping equivalence, novelty,
universal identity, consciousness, human alignment, independent science or
production. Stronger scientific acceptance needs the configured independent
authority; production has a separate authorization gate. Neither is needed
for this bounded local experiment.
