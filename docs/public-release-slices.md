## Public release readiness

Explicit public-release readiness is authorized under named state slice
`public-release-readiness-v1`: additive `LICENSE`, `NOTICE`, `SECURITY.md`,
`CONTRIBUTING.md`, `RESEARCH.md`, `scripts/run_public_tests.sh`, README
navigation, replacement of machine-specific paths in `README.md`, `NOTES.md`, and
`docs/qwen38-local-assessment.md` with environment variables, and a CI workflow.
It changes no experiment source, frozen protocol, threshold, result, or claim
ceiling and authorizes no model execution or claim escalation. Every mutation in
this phase names state slice `public-release-readiness-v1`.

Explicit public README documentation is authorized under named state slice
`public-release-readme-v2`: documentation-only rewrite of `README.md` (summary of
findings, verification and CI/CD acceptance record, quick start) with the
original documentation preserved. It adds no CI file, source change, evidence,
or claim-ceiling change and authorizes no execution, provider call, spend,
accepted Evidence Ledger mutation, or Level2+ claim. Every mutation in this
phase names state slice `public-release-readme-v2`.
