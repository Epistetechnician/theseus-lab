# V39 first-MLP fit-factorial V2

2026-10-07. State slices `theseus-lab-sandbox-v1`,
`theseus-v39-first-mlp-fit-factorial-v2`, and
`global-experiment-assurance-lifecycle-v1`.

## V1 preflight disposition

The V1 source packet froze, but its first verification call targeted the V39
adapter instead of the shared V36 runner object that owns `verify_source`.
Both runtime preflights stopped before runtime qualification, artifact-root
creation, prompt generation, model forward, or training. V1 has no scientific
result and remains preserved as a preflight stop. V2 binds the correction
through `r.verify_source()` and includes the exact V1 source, freeze, tests,
protocol, and closeout bytes as historical inputs.

## Frozen experiment

V2 continues the same narrowly scoped pilot: one reused
`mrm8488/spanish-gpt2` checkpoint at revision
`fc396ce08c65ea334dcf541de7dc05cac89a29b2`, MLP site 0, three candidate seeds,
and a 2×2 design crossing hidden expansion 2×/4× with 1,024/4,096 updates.
The objective calls the saved V36 core fitter directly because the V39 wrapper
expects padded 3D captures while this pilot supplies already pooled 2D token
vectors. This preserves the original objective without a second pooling pass.

All paired thresholds, fixture rules, controls, resource caps, and claim
ceilings remain as declared in the V1 protocol document. V2 uses a new packet
digest, a separate artifact root, and this exact source set. The freeze occurs
before new adaptation or challenge fixtures are generated.

## Claim ceiling

`LOCAL_REUSED_GPT2_FIRST_MLP_FIT_FACTOR_DIAGNOSIS_V2_ONLY`. This is one reused
guide and one local MLP. It does not establish architecture-general validity,
whole-model mechanistic identity, consciousness, human-alignment equivalence,
independent scientific acceptance, or production readiness. Reviewer status
is `NOT_SUBMITTED`; production remains a separate authorization gate.
