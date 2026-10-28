# Lean proof: effect-geometry bound

Kernel-checked (Lean 4.30.0, Mathlib v4.30.0): for a predicted effect `p` and observed effect `o` in a real inner product space with `cos = <p,o> / (||p|| ||o||)`,

`||o|| * sqrt(1 - cos^2) <= ||p - o||`

(`norm_mul_sin_le`, plus squared forms). Axioms: `propext`, `Classical.choice`, `Quot.sound`. No `sorry`.

Use in the theseus-lab analysis: a small normalized error bounds only `||o|| sin(theta)`, so when the observed effect norm is tiny (about 1e-6 of the logit norm in Band A) a small error does not imply a high cosine. This is why the median-cosine gate was the binding gate. It is a statement about vectors, not about the experiments or floating-point behaviour.

## Check

```sh
cd formal/lean && lake exe cache get && lake build
```
