# Causal-effect identifiability boundary

State slices: `theseus-lab-sandbox-v1` and
`theseus-causal-effect-identifiability-boundary-v1`.

Base revision: `202f459d6d89df9a038d78d58e0af825d287fcd8` (the worktree was
already dirty; this note is a new, standalone file).

## Claim

Matching a replacement's outputs on finitely many observed inputs does not,
by itself, identify its response to an unseen intervention. One sufficient
way to bound causal-effect error is to control approximation error at both
the factual input and the counterfactual input, together with a bound on the
shared downstream map.

This is a formal boundary on inference from finite observations. It is not a
proof of Theseus continuity, a result about consciousness, or a claim that a
particular neural-network architecture realizes the counterexample below.

## Conditional endpoint bound

Let `m, g : Z -> U` be the original and replacement modules. Let
`D : U -> Y` be the same downstream computation for both modules, and suppose
`D` is `L`-Lipschitz on the relevant outputs. For an input `x` and an
intervention `delta`, define

```text
C_m(x, delta) = D(m(x + delta)) - D(m(x))
C_g(x, delta) = D(g(x + delta)) - D(g(x)).
```

If both endpoint errors are bounded by `epsilon`,

```text
||g(x) - m(x)|| <= epsilon
||g(x + delta) - m(x + delta)|| <= epsilon,
```

then

```text
||C_g(x, delta) - C_m(x, delta)|| <= 2 L epsilon.
```

**Proof.** Add and subtract the two original downstream values and apply the
triangle inequality:

```text
||C_g - C_m||
 <= ||D(g(x + delta)) - D(m(x + delta))||
    + ||D(g(x)) - D(m(x))||
 <= L ||g(x + delta) - m(x + delta)||
    + L ||g(x) - m(x)||
 <= 2 L epsilon.
```

For identity downstream `D`, take `L = 1`. This is an absolute error bound;
it does not imply a small relative error when the original causal effect is
near zero. A resolvability rule is still needed for relative-effect metrics.

The result is conditional on a uniform endpoint-error bound over the inputs
being claimed. Low average error on a finite sample does not establish that
uniform premise for unseen inputs.

## Finite-observation counterexample

Let `S = {s_1, ..., s_n}` be any finite set of observed scalar inputs and
choose an intervention endpoint `z*` outside `S`. Define the smooth function

```text
h(z) = c * product_i ((z - s_i) / (z* - s_i))
```

for any nonzero scalar `c`. Let the reference module be `m(z) = 0` and the
replacement be `g(z) = h(z)`. Then `g(s_i) = m(s_i) = 0` at every observed
input, while `g(z*) = c`.

Choose any observed factual input `x` and an intervention `delta = z* - x`.
The reference effect is zero, but the replacement effect is `c`. Thus exact
agreement on every observed input does not determine the response at an
unobserved intervention endpoint, even within smooth functions. Vector-valued
outputs have the same construction by multiplying `h` by any fixed nonzero
output vector.

The counterexample is a non-identifiability result for unrestricted smooth
functions and finite observations. Architecture constraints, derivative
bounds, or other verified structural assumptions can narrow the function
class, but those assumptions must be stated and checked separately.

## Consequence for replacement evidence

Unperturbed behavior checks evaluate candidate outputs on ordinary inputs.
Causal-effect checks compare paired outputs at factual and intervened inputs.
Passing the former does not imply the latter. The sufficient bound above
requires endpoint control, and other bounds would need their own assumptions.
V16 and V18 preserved behavior while missing causal gates
([V16 results](replacement-loss-v16-results.md),
[V18 results](replacement-loss-v18-results.md)); V17 found an inconsistent
causal/value tradeoff ([V17 results](replacement-loss-v17-results.md)). These
results are consistent with that separation, but do not show that endpoint
coverage caused the failures. V19 selected no development candidate
([V19 results](replacement-loss-v19-development-results.md)). V20B's
development behavior gates passed 9/9 cells while its causal gate passed 3/9;
all ordinary and compositional effect ratios exceeded `1.06`, and no
candidate qualified ([V20B results](replacement-loss-v20b-development-results.md)).
V20B stopped before confirmation challenges.

For a future learned-replacement claim, the frozen packet should therefore:

1. Define the intervention distribution and both endpoint populations before
   candidate fitting.
2. Separate fit, validation, and held-out intervention endpoints by prompt.
3. Freeze candidate predictions before revealing held-out intervention
   outcomes.
4. Report factual-output error, counterfactual-output error, absolute causal
   effect error, relative causal effect error, and ordinary behavior as
   separate measures.
5. Keep any uniform or smoothness-based generalization claim conditional on
   an explicit, independently checked bound; empirical averages alone do
   not provide it.

Finite experiments can support a bounded predictive claim over a stated
distribution. A claim of equality for every input requires a structural
argument about the function class or a formal verification of the concrete
implementation; more sampled prompts alone cannot establish it.

## Evidence and claim ceiling

- This artifact is a mathematical derivation, not a model experiment.
- Experimental data and model artifacts were not opened or generated for this
  note; prior prompts, captures, and candidate tensors remain in external
  owner custody.
- Assurance for the derivation: not independently reviewed.
- Maximum claim ceiling:
  `FORMAL_CONDITIONAL_CAUSAL_EFFECT_BOUND_AND_FINITE_OBSERVATIONAL_NONIDENTIFIABILITY`.
- Nonclaims: novelty priority, exact equivalence of any learned replacement,
  architecture-general learned mechanism preservation, consciousness,
  alignment, independent scientific acceptance, and production readiness.
- Next scientific gate: a newly frozen learned-replacement protocol with
  explicit endpoint coverage, followed by held-out causal prediction and a
  separate blind replication. Any independent-science claim still requires
  an external reviewer with independent authority.
