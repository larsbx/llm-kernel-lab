import SpeculativeSampling

/-!
# Independently restated acceptance obligations

These wrappers deliberately spell out the public propositions. They prevent
acceptance from relying only on declaration names or successful compilation of
a theorem whose statement has been weakened. No equal-total, positive-token,
unique-row, or nonzero-rejection assumptions are introduced.

The interface certifies the Lean finite-mass model. Runtime conformance,
uniform/independent randomness, floating-point logits, and model acceleration
remain separate claims.
-/

namespace SpeculativeSamplingContract

open SpeculativeSampling

theorem target_identity (weights : List (Nat × Nat)) (w : Nat × Nat)
    (hw : w ∈ weights) :
    outputMass weights w * targetTotal weights = outputSpace weights * w.1 :=
  output_mass_target_identity weights w hw

theorem mass_total (weights : List (Nat × Nat)) :
    sumBy (outputMass weights) weights = outputSpace weights :=
  output_mass_total weights

theorem rejection_identity (weights : List (Nat × Nat)) :
    sumBy (residual weights) weights = sumBy (rejected weights) weights :=
  residual_equals_rejection weights

theorem positive_space (weights : List (Nat × Nat))
    (hp : 0 < sumBy Prod.fst weights) (hq : 0 < sumBy Prod.snd weights) :
    0 < outputSpace weights :=
  output_space_positive weights ⟨hp, hq⟩

theorem common_denominator (weights : List (Nat × Nat)) :
    sumBy (scaledTarget weights) weights = targetTotal weights * draftTotal weights ∧
    sumBy (scaledDraft weights) weights = targetTotal weights * draftTotal weights :=
  ⟨scaled_target_total weights, scaled_draft_total weights⟩

theorem complete_contract (weights : List (Nat × Nat))
    (hp : 0 < sumBy Prod.fst weights) (hq : 0 < sumBy Prod.snd weights) :
    0 < outputSpace weights ∧
    sumBy (outputMass weights) weights = outputSpace weights ∧
    ∀ w ∈ weights,
      outputMass weights w * targetTotal weights = outputSpace weights * w.1 :=
  exact_target_distribution weights ⟨hp, hq⟩

#print axioms target_identity
#print axioms mass_total
#print axioms rejection_identity
#print axioms positive_space
#print axioms common_denominator
#print axioms complete_contract

end SpeculativeSamplingContract
