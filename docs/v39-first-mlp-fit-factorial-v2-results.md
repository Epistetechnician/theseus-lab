# V39 first-MLP fit-factorial V2 results

2026-10-07, revision `202f459d6d89df9a038d78d58e0af825d287fcd8`, state slices
`theseus-lab-sandbox-v1`, `theseus-v39-first-mlp-fit-factorial-v2`, and
`global-experiment-assurance-lifecycle-v1`.

**INCONCLUSIVE — terminal pre-fit shape error.** V2 source verification,
runtime fingerprinting, and five focused tests passed in both isolated
runtimes. The primary then loaded the frozen V39 GPT-2 checkpoint and generated
one seed's 128 adaptation plus 128 donor prompts. The V39 marked-capture path
appended a mask feature to each 768-wide activation. The pilot's pooling code
did not remove it, so the fitter received width 769 and failed against the
guide's width 768.

The run completed 256 capture forwards and zero optimizer updates. Candidate
initialization did not complete. No held-out challenge fixtures, candidate
predictions, guide-oracle evaluations, or replica run occurred. The failed
root is preserved; V2 will not be retried. Its prompt and token identities are
excluded from the new V3 fixture stream.

Artifact custody is `PrimaryED`; the runtime is on `Backup ED`. Paid execution
was $0. Energy and full economic cost were not measured. Reviewer status is
`NOT_SUBMITTED`; reproduction status is `NOT_RUN_AFTER_PRIMARY_PREFIT_STOP`.
This is an execution stop, not a scientific FAIL or PASS.

Claim ceiling `LOCAL_REUSED_GPT2_FIRST_MLP_FIT_FACTOR_DIAGNOSIS_V2_ONLY`.
No conclusion is drawn about replacement quality, capacity, or optimization.

V2 packet SHA-256:
`f8d455412f6be813be91acf3bff3cb30cf6026bede98624e5072bd23d98078b7`.
External preflight receipt: `theseus-lab-sandbox-v1-v39-first-mlp-fit-factorial-v2/primary/preflight-stop.json`.
