import Mathlib

/-!
# Effect-geometry bound used in the theseus-lab Band A analysis

For a predicted effect `p` and an observed effect `o` in a real inner product space, with
`cosθ = ⟪p,o⟫ / (‖p‖‖o‖)`, the residual `p - o` satisfies
`‖o‖² (1 - cos²θ) ≤ ‖p - o‖²`, i.e. `‖o‖·sinθ ≤ ‖p - o‖`.

Consequence for the cosine gate: a tiny normalized error does not imply a high cosine,
because `‖o‖` itself may be tiny; the residual only bounds `‖o‖ sinθ`.
-/

open RealInnerProductSpace

namespace EvidenceLean

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- `‖o‖² (1 - cos²) ≤ ‖p - o‖²`, written without division by the cosine. -/
theorem sin_sq_mul_norm_sq_le (p o : E) (hp : p ≠ 0) :
    ‖o‖ ^ 2 - ⟪p, o⟫ ^ 2 / ‖p‖ ^ 2 ≤ ‖p - o‖ ^ 2 := by
  have hp2 : 0 < ‖p‖ ^ 2 := by positivity
  have hsub : ‖p - o‖ ^ 2 = ‖p‖ ^ 2 - 2 * ⟪p, o⟫ + ‖o‖ ^ 2 := norm_sub_sq_real p o
  rw [hsub]
  have key : 0 ≤ (‖p‖ ^ 2 - ⟪p, o⟫) ^ 2 / ‖p‖ ^ 2 := by positivity
  have expand : (‖p‖ ^ 2 - ⟪p, o⟫) ^ 2 / ‖p‖ ^ 2
      = ‖p‖ ^ 2 - 2 * ⟪p, o⟫ + ⟪p, o⟫ ^ 2 / ‖p‖ ^ 2 := by
    field_simp
    ring
  linarith [key, expand]

/-- The same bound in cosine form, for nonzero `p` and `o`. -/
theorem norm_mul_sin_sq_le (p o : E) (hp : p ≠ 0) (ho : o ≠ 0) :
    ‖o‖ ^ 2 * (1 - (⟪p, o⟫ / (‖p‖ * ‖o‖)) ^ 2) ≤ ‖p - o‖ ^ 2 := by
  have hp0 : ‖p‖ ≠ 0 := norm_ne_zero_iff.mpr hp
  have ho0 : ‖o‖ ≠ 0 := norm_ne_zero_iff.mpr ho
  have h := sin_sq_mul_norm_sq_le p o hp
  have eq : ‖o‖ ^ 2 * (1 - (⟪p, o⟫ / (‖p‖ * ‖o‖)) ^ 2)
      = ‖o‖ ^ 2 - ⟪p, o⟫ ^ 2 / ‖p‖ ^ 2 := by
    field_simp
  rw [eq]
  exact h

/-- Square-root form: `‖o‖ · sinθ ≤ ‖p - o‖`, where `sinθ = √(1 - cos²θ)`. -/
theorem norm_mul_sin_le (p o : E) (hp : p ≠ 0) (ho : o ≠ 0) :
    ‖o‖ * Real.sqrt (1 - (⟪p, o⟫ / (‖p‖ * ‖o‖)) ^ 2) ≤ ‖p - o‖ := by
  have h := norm_mul_sin_sq_le p o hp ho
  have hnn : 0 ≤ ‖p - o‖ := norm_nonneg _
  have hmul : ‖o‖ * Real.sqrt (1 - (⟪p, o⟫ / (‖p‖ * ‖o‖)) ^ 2)
      = Real.sqrt (‖o‖ ^ 2 * (1 - (⟪p, o⟫ / (‖p‖ * ‖o‖)) ^ 2)) := by
    rw [Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (norm_nonneg _)]
  rw [hmul]
  calc Real.sqrt (‖o‖ ^ 2 * (1 - (⟪p, o⟫ / (‖p‖ * ‖o‖)) ^ 2))
      ≤ Real.sqrt (‖p - o‖ ^ 2) := Real.sqrt_le_sqrt h
    _ = ‖p - o‖ := Real.sqrt_sq hnn

end EvidenceLean

#print axioms EvidenceLean.sin_sq_mul_norm_sq_le
#print axioms EvidenceLean.norm_mul_sin_sq_le
#print axioms EvidenceLean.norm_mul_sin_le
