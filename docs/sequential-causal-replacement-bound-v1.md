# Sequential replacement bound for causal effects

State slices: `theseus-lab-sandbox-v1`,
`theseus-sequential-causal-replacement-bound-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Claim

For a deterministic chain, bounded local replacement error at each stage,
measured on inputs produced by the already-replaced prefix, gives a bound on
the final output error. If the same bound holds for both factual and
intervened evaluations, it also bounds the error in the causal effect.

This is a conditional composition result. It does not show that a learned
replacement satisfies its local-error assumptions, or that finite training
data imply those assumptions on unseen inputs.

## Theorem

Let the original chain be

```text
F = f_n o f_(n-1) o ... o f_1
```

and the replacement chain be

```text
G = g_n o g_(n-1) o ... o g_1
```

where `f_i` and `g_i` have the same input and output spaces. For each query
`q` in a declared set `Q`, define the stage-i input under the replaced prefix
as

```text
z_i(q) = g_(i-1)(...g_1(q)...),   z_1(q) = q.
```

Assume that for every `q` in `Q`:

1. `||g_i(z_i(q)) - f_i(z_i(q))|| <= epsilon_i` for each stage `i`;
2. each original downstream stage `f_j`, for `j > i`, is `L_j`-Lipschitz
   over the intermediate values used in the comparison.

Then

```text
||G(q) - F(q)|| <= E
E = sum_(i=1)^n epsilon_i * product_(j=i+1)^n L_j,
```

with an empty product equal to `1`.

For two queries `q0` and `q1`, define their effects as

```text
C_F = F(q1) - F(q0)
C_G = G(q1) - G(q0).
```

If the premises hold for both queries, then

```text
||C_G - C_F|| <= 2E.
```

If additionally `||C_F|| >= rho > 0`, the relative effect error is at most
`2E / rho`.

## Proof

Define hybrid chains `H_i` by replacing the first `i` stages and leaving the
remaining stages original. Thus `H_0 = F`, `H_n = G`, and

```text
G(q) - F(q) = sum_(i=1)^n (H_i(q) - H_(i-1)(q)).
```

The input to stage `i` is `z_i(q)` in both adjacent hybrids because their
prefixes are the same. Their stage-i outputs differ by at most
`epsilon_i`. The remaining stages are the original maps
`f_(i+1), ..., f_n`; their Lipschitz constants bound the propagated
difference by `epsilon_i * product_(j=i+1)^n L_j`. The triangle inequality
over the hybrid differences gives the stated bound `E`.

For the effect, add and subtract the original outputs at both endpoints:

```text
||C_G - C_F||
<= ||G(q1) - F(q1)|| + ||G(q0) - F(q0)||
<= E + E = 2E.
```

Dividing by `||C_F|| >= rho` gives the relative-error bound.

## Assumptions that matter in an experiment

- `Q` must include both factual and counterfactual endpoint evaluations.
  Reusing only ordinary factual inputs does not establish the premise for an
  intervention endpoint.
- At stage `i`, local error must be measured on inputs from the already-
  replaced prefix. Measuring it only on the original guide's prefix inputs
  does not establish this theorem's premise when earlier replacements shift
  the activation distribution.
- The Lipschitz bounds must hold on all intermediate values traversed by the
  hybrid comparisons. A finite empirical maximum is evidence only for those
  measured values unless a separate uniform bound is proved.
- The interventions must have the same operational meaning in `F` and `G`.
  Routing changes, state, randomness, clipping, normalization, or different
  intervention hooks require an expanded computation model.
- Small absolute effect error does not guarantee small relative error for a
  near-zero original effect; the lower bound `rho` is needed.

For an exact additive split, every `epsilon_i` is zero at the split site, so
the bound reduces to exact output and effect equality. Learned replacements
need not have zero local error, and this theorem does not say they can be
learned from finite observations.

The inequality follows from telescoping and Lipschitz composition. This note
does not claim the bound itself is novel.

## Use in a future replacement ladder

Report per-stage factual and intervened local errors, the inputs on which
they were measured, and any downstream sensitivity bounds. Keep the
resulting endpoint/effect bound separate from empirical held-out metrics.
If the sensitivity or uniform-error premises cannot be certified, report
the bound as unavailable; do not substitute average training loss for it.

A future learned-replacement experiment still needs a distinct frozen
hypothesis, explicit intervention semantics, a lawful and fixed task corpus,
and predictions sealed before held-out outcomes are opened. Existing V3,
V17, V28, and V30 outcomes remain unchanged. This theorem does not upgrade
their results or establish a faithful Network of Theseus comparison.

## Evidence and claim ceiling

- This artifact is a mathematical derivation, not a model experiment.
- No checkpoint, prompt, fixture, or external-custody artifact was opened or
  generated for this derivation.
- Assurance: `INTERNAL_PASS` for the derivation and documentation checks only;
  it has not been independently reviewed. Reviewer status: `NOT_SUBMITTED`.
- Reproducibility: `NOT_APPLICABLE`.
- Maximum claim ceiling:
  `FORMAL_CONDITIONAL_SEQUENTIAL_OUTPUT_AND_CAUSAL_EFFECT_ERROR_BOUND`.
- Nonclaims: novelty priority, learned-replacement success, generalization to
  unseen interventions, architecture-general Theseus continuity,
  consciousness, alignment, independent scientific acceptance, and
  production readiness.
- Next blocked gate: a separately frozen learned-replacement packet with an
  explicit task/corpus and intervention definition, followed by held-out
  prediction. Stronger independent-science claims still require blind
  replication and an authorized external reviewer.
