# Research report and findings

Question (the Ship of Theseus thesis): can a system keep its observed behaviour
while its explicitly tracked computational mechanism is replaced, split, or
lost, and how would we know?

Everything here is a **local, owner-run observation** on small open checkpoints
(GPT-2, distilgpt2, OPT-125M, SmolLM/SmolLM2-135M) plus one Qwen assessment
slice. Nothing was independently reviewed (`Reviewer: NOT_SUBMITTED`). The most
complete synthesis is `docs/theseus-all-runs-audit-v1.md`; per-run protocols,
freezes, and results are under `docs/`; the running journal is `NOTES.md`.

## Findings

**1. Behaviour and mechanism can come apart (the thesis, in its weak form).**
In a deterministic state machine a structural split keeps visible and held-out
behaviour at 1.0 while mechanism continuity falls to 0.4167; a replacement that
matches visible probes fails held-out probes, so visible equivalence can be a
probe-coverage artefact.

**2. Raw labels and parameter positions are not stable causal identity.** In
GPT-2, a function-preserving MLP permutation changed 100% of the selected
positions yet preserved top-1 behaviour (mean relative logit error 3e-7). The
same-label ablation then looked unrelated (cosine -0.983) while the
inverse-mapped ablation matched (0.9999996). Mechanism claims have to be
representation-relative. A PyTorch/Transformers reference matched the NumPy
implementation (error 4.9e-7); 48 permutation cases passed.

**3. Exact structural replacement is easy; learned replacement is hard.**
Splitting a trained MLP channel into two children (exact algebra) passed 6/6
across OPT-125M and SmolLM2-135M, with all 12 wrong-source/random controls
failing as required. This is a constructed positive control, not independent
retraining.

**4. Only a narrow learned replacement worked.** V8: an independently fitted
affine output projection reproduced in 9/9 fresh checkpoint-by-seed cases
(strongest bounded positive result). V28 repeated the treatment result on fresh
targets but failed a frozen numeric-tolerance control gate, so the recorded
outcome is FAIL.

**5. Whole nonlinear blocks did not replicate under the frozen gates.**
Whole-MLP replacement showed high activation similarity yet failed causal
prediction 9/9 (V16); causal/refinement objectives kept behaviour but failed
module and downstream-effect gates (V17/V18); no qualified candidate in V19,
V20/V20B, V21/V21B; relative gains over affine did not give absolute retention
(V14: 0/12). V31: random and zero blocks retained 100% task accuracy, i.e. the
target was not behaviourally indispensable. V9 ladder: 21/27 stages passed,
overall FAIL.

**6. Diagnosing the V39 failure.** Adaptation-only diagnostics found that the
fitted module matches values (nRMSE about 0.10-0.12) but not local derivatives
(Jacobian nRMSE 0.88-0.89), with large error even along the training
directions (0.83-0.89): derivative underfitting. A 2x2 factorial (width x
updates) showed more optimizer updates helped (22% lower JVP error) while extra
width did not clear its frozen bar (12% lower, needed 20%); no cell passed the
held-out retention rule. V38 showed final-token-only fitting leaves prefix
positions badly fit (nRMSE 0.40-0.67 vs 0.014-0.020).

**7. The Qwen behavioural safeguard slice is a measurement-validity result.**
A lexical refusal proxy matched 145/939 responses (15.4%, Wilson 95% CI
13.3-17.9%) against a 90% target. A separate assessor scored 53.6% effective
safe (CI 50.4-56.7%) on the harmful set and flagged 62% of benign contrast
prompts unsafe. Both locked targets failed. The two instruments do not give a
calibrated verdict, and a second independent evaluator is absent (blocked), so
no alignment conclusion follows.

**8. Process findings.** Many runs ended in setup, identity, or runtime-binding
stops (cloud V4C-V4K, V15, V22-V29) before any scientific outcome; these are
recorded as stops, not as results. Outcome-aware variants are never combined
with the original holdout.

## What this does not establish

No independent replication, no human-calibrated alignment result, no claim of
consciousness or mechanistic understanding, no architecture-general or
whole-mechanism replacement result, and no novelty claim: progressive
conversion is prior art (Network of Theseus, arXiv:2512.04198) and aligned
intervention distillation is prior art (Causal Distillation, NAACL 2022).

## Reproduce

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHON=.venv/bin/python scripts/run_public_tests.sh
```

Each test module runs in its own process. Three modules need the owner's frozen
runtime pins or private custody directory and are reported as
`SKIPPED-OWNER-ENV` (set `STRICT=1` to treat them as failures). Model
checkpoints, assessment corpora, and raw transcripts are held in owner custody
and are not in this repository; commands that use them take a `$THESEUS_CUSTODY`
root.
