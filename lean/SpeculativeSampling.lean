import Std

/-!
# Exact speculative sampling: finite integer mass kernel

Inputs are aligned `(target weight, draft weight)` pairs. The two totals may
be different. All arithmetic is over unbounded natural numbers.

`accepted`, `residual`, and `rejected` are executable functions. The public
theorems establish their conservation laws and the target-distribution identity
for the two-stage ticket-mass construction. In the rejection-free case the
ticket space has size `P * Q`; otherwise it has size `P * Q * R`.

This file does not verify a Python interval decoder, a random-number generator,
a compiled binary, an entire autoregressive decoder, or a performance claim.
Its probability statement is an exact equality of finite integer masses,
conditional on uniform tickets and the stated two-stage interpretation.
-/

namespace SpeculativeSampling

abbrev Weights := List (Nat × Nat)

/-- A transparent finite sum, independent of mathlib. -/
def sumBy {α : Type} (f : α → Nat) : List α → Nat
  | [] => 0
  | x :: xs => f x + sumBy f xs

private theorem sumBy_add {α : Type} (f g : α → Nat) (xs : List α) :
    sumBy (fun x => f x + g x) xs = sumBy f xs + sumBy g xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
      simp [sumBy, ih, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]

private theorem sumBy_mul_right {α : Type} (f : α → Nat) (k : Nat)
    (xs : List α) : sumBy (fun x => f x * k) xs = sumBy f xs * k := by
  induction xs with
  | nil => simp [sumBy]
  | cons x xs ih => simp [sumBy, ih, Nat.add_mul]

private theorem sumBy_mul_left {α : Type} (f : α → Nat) (k : Nat)
    (xs : List α) : sumBy (fun x => k * f x) xs = k * sumBy f xs := by
  simpa [Nat.mul_comm] using sumBy_mul_right f k xs

private theorem sumBy_congr {α : Type} (f g : α → Nat) (xs : List α)
    (h : ∀ x, f x = g x) : sumBy f xs = sumBy g xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [sumBy, h x, ih]

private theorem sumBy_member_le {α : Type} (f : α → Nat)
    (xs : List α) (x : α) (hx : x ∈ xs) : f x ≤ sumBy f xs := by
  induction xs with
  | nil => simp at hx
  | cons y ys ih =>
      simp only [List.mem_cons] at hx
      cases hx with
      | inl h =>
          subst x
          exact Nat.le_add_right _ _
      | inr h =>
          exact Nat.le_trans (ih h) (Nat.le_add_left _ _)

def targetTotal (weights : Weights) : Nat := sumBy Prod.fst weights
def draftTotal (weights : Weights) : Nat := sumBy Prod.snd weights
def denominator (weights : Weights) : Nat := targetTotal weights * draftTotal weights

/-- Zero entries are allowed, but each distribution must have positive total. -/
def Valid (weights : Weights) : Prop :=
  0 < targetTotal weights ∧ 0 < draftTotal weights

def scaledTarget (weights : Weights) (w : Nat × Nat) : Nat :=
  w.1 * draftTotal weights

def scaledDraft (weights : Weights) (w : Nat × Nat) : Nat :=
  w.2 * targetTotal weights

def accepted (weights : Weights) (w : Nat × Nat) : Nat :=
  min (scaledTarget weights w) (scaledDraft weights w)

def residual (weights : Weights) (w : Nat × Nat) : Nat :=
  scaledTarget weights w - accepted weights w

def rejected (weights : Weights) (w : Nat × Nat) : Nat :=
  scaledDraft weights w - accepted weights w

def totalAccepted (weights : Weights) : Nat := sumBy (accepted weights) weights
def totalResidual (weights : Weights) : Nat := sumBy (residual weights) weights
def totalRejected (weights : Weights) : Nat := sumBy (rejected weights) weights

/-- Executable mass row; no stochastic operation is hidden in this record. -/
structure MassRow where
  target : Nat
  draft : Nat
  accept : Nat
  remainder : Nat
  reject : Nat
  deriving Repr, DecidableEq

def compileRow (weights : Weights) (w : Nat × Nat) : MassRow :=
  { target := scaledTarget weights w
    draft := scaledDraft weights w
    accept := accepted weights w
    remainder := residual weights w
    reject := rejected weights w }

def compile (weights : Weights) : List MassRow := weights.map (compileRow weights)

theorem compile_length (weights : Weights) : (compile weights).length = weights.length := by
  simp [compile]

theorem accepted_le_target (weights : Weights) (w : Nat × Nat) :
    accepted weights w ≤ scaledTarget weights w := Nat.min_le_left _ _

theorem accepted_le_draft (weights : Weights) (w : Nat × Nat) :
    accepted weights w ≤ scaledDraft weights w := Nat.min_le_right _ _

theorem accepted_add_residual (weights : Weights) (w : Nat × Nat) :
    accepted weights w + residual weights w = scaledTarget weights w := by
  simpa [residual, Nat.add_comm] using
    Nat.sub_add_cancel (accepted_le_target weights w)

theorem accepted_add_rejected (weights : Weights) (w : Nat × Nat) :
    accepted weights w + rejected weights w = scaledDraft weights w := by
  simpa [rejected, Nat.add_comm] using
    Nat.sub_add_cancel (accepted_le_draft weights w)

theorem scaled_target_total (weights : Weights) :
    sumBy (scaledTarget weights) weights = denominator weights := by
  exact sumBy_mul_right Prod.fst (draftTotal weights) weights

theorem scaled_draft_total (weights : Weights) :
    sumBy (scaledDraft weights) weights = denominator weights := by
  calc
    sumBy (scaledDraft weights) weights = draftTotal weights * targetTotal weights :=
      sumBy_mul_right Prod.snd (targetTotal weights) weights
    _ = denominator weights := Nat.mul_comm _ _

theorem accepted_residual_total (weights : Weights) :
    totalAccepted weights + totalResidual weights = denominator weights := by
  calc
    totalAccepted weights + totalResidual weights =
        sumBy (fun w => accepted weights w + residual weights w) weights :=
      (sumBy_add (accepted weights) (residual weights) weights).symm
    _ = sumBy (scaledTarget weights) weights :=
      sumBy_congr _ _ weights (accepted_add_residual weights)
    _ = denominator weights := scaled_target_total weights

theorem accepted_rejected_total (weights : Weights) :
    totalAccepted weights + totalRejected weights = denominator weights := by
  calc
    totalAccepted weights + totalRejected weights =
        sumBy (fun w => accepted weights w + rejected weights w) weights :=
      (sumBy_add (accepted weights) (rejected weights) weights).symm
    _ = sumBy (scaledDraft weights) weights :=
      sumBy_congr _ _ weights (accepted_add_rejected weights)
    _ = denominator weights := scaled_draft_total weights

theorem residual_equals_rejection (weights : Weights) :
    totalResidual weights = totalRejected weights := by
  exact Nat.add_left_cancel
    ((accepted_residual_total weights).trans (accepted_rejected_total weights).symm)

theorem residual_zero_of_total_zero (weights : Weights) (w : Nat × Nat)
    (hw : w ∈ weights) (hzero : totalResidual weights = 0) :
    residual weights w = 0 := by
  have hle := sumBy_member_le (residual weights) weights w hw
  change residual weights w ≤ totalResidual weights at hle
  rw [hzero] at hle
  exact Nat.eq_zero_of_le_zero hle

/--
Equiprobable finite ticket-space size. With rejection, a residual ticket is
included even for accepted proposals, where it is ignored. This gives a single
uniform product space without dividing by zero when rejection is impossible.
-/
def outputSpace (weights : Weights) : Nat :=
  if totalResidual weights = 0 then denominator weights
  else denominator weights * totalResidual weights

/--
Mass of output token `w` in the two-stage ticket model. Accepted paths contribute
`accepted * R`; rejected proposals contribute `totalRejected * residual`.
-/
def outputMass (weights : Weights) (w : Nat × Nat) : Nat :=
  if totalResidual weights = 0 then accepted weights w
  else accepted weights w * totalResidual weights +
    totalRejected weights * residual weights w

theorem output_mass_closed_form (weights : Weights) (w : Nat × Nat)
    (hw : w ∈ weights) :
    outputMass weights w =
      if totalResidual weights = 0 then scaledTarget weights w
      else scaledTarget weights w * totalResidual weights := by
  by_cases hzero : totalResidual weights = 0
  · have hz := residual_zero_of_total_zero weights w hw hzero
    have ha := accepted_add_residual weights w
    simp only [hz, Nat.add_zero] at ha
    simpa [outputMass, hzero] using ha
  · simp only [outputMass, hzero, ↓reduceIte]
    calc
      accepted weights w * totalResidual weights +
          totalRejected weights * residual weights w =
          accepted weights w * totalResidual weights +
          residual weights w * totalResidual weights := by
            rw [← residual_equals_rejection weights]
            rw [Nat.mul_comm (totalResidual weights) (residual weights w)]
      _ = (accepted weights w + residual weights w) * totalResidual weights :=
        (Nat.add_mul _ _ _).symm
      _ = scaledTarget weights w * totalResidual weights := by
        rw [accepted_add_residual]

/-- Universal exact target law, expressed without rational division. -/
theorem output_mass_target_identity (weights : Weights) (w : Nat × Nat)
    (hw : w ∈ weights) :
    outputMass weights w * targetTotal weights = outputSpace weights * w.1 := by
  rw [output_mass_closed_form weights w hw]
  by_cases hzero : totalResidual weights = 0
  · simp [outputSpace, hzero, scaledTarget, denominator,
      Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  · simp [outputSpace, hzero, scaledTarget, denominator,
      Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]

theorem output_mass_total (weights : Weights) :
    sumBy (outputMass weights) weights = outputSpace weights := by
  change sumBy (fun w => outputMass weights w) weights = outputSpace weights
  by_cases hzero : totalResidual weights = 0
  · have ha := accepted_residual_total weights
    simp only [hzero, Nat.add_zero] at ha
    simpa [outputMass, outputSpace, hzero, totalAccepted] using ha
  · simp only [outputMass, outputSpace, hzero, ↓reduceIte]
    rw [sumBy_add, sumBy_mul_right, sumBy_mul_left]
    change totalAccepted weights * totalResidual weights +
      totalRejected weights * totalResidual weights =
      denominator weights * totalResidual weights
    rw [← residual_equals_rejection weights, ← Nat.add_mul,
      accepted_residual_total]

theorem denominator_positive (weights : Weights) (hvalid : Valid weights) :
    0 < denominator weights := Nat.mul_pos hvalid.1 hvalid.2

theorem output_space_positive (weights : Weights) (hvalid : Valid weights) :
    0 < outputSpace weights := by
  by_cases hzero : totalResidual weights = 0
  · simpa [outputSpace, hzero] using denominator_positive weights hvalid
  · simpa [outputSpace, hzero] using
      Nat.mul_pos (denominator_positive weights hvalid) (Nat.pos_of_ne_zero hzero)

/--
The complete finite-mass contract: nonempty uniform space, total mass one after
normalization, and each token's mass equal to its normalized target weight.
Zero token weights and unequal target/draft totals are permitted.
-/
theorem exact_target_distribution (weights : Weights) (hvalid : Valid weights) :
    0 < outputSpace weights ∧
    sumBy (outputMass weights) weights = outputSpace weights ∧
    ∀ w ∈ weights,
      outputMass weights w * targetTotal weights = outputSpace weights * w.1 := by
  exact ⟨output_space_positive weights hvalid, output_mass_total weights,
    fun w hw => output_mass_target_identity weights w hw⟩

-- Kernel-reduced checks exercise unequal totals, missing draft support, and the
-- zero-rejection branch. They supplement, and do not replace, the universal laws.
example : totalResidual [(1, 1), (1, 3)] = 2 := by decide
example : totalResidual [(1, 2), (2, 4)] = 0 := by decide
example : totalResidual [(1, 0), (2, 1), (0, 3)] = 9 := by decide
example : outputMass [(1, 0), (2, 1), (0, 3)] (1, 0) = 36 := by decide

#print axioms accepted_add_residual
#print axioms accepted_add_rejected
#print axioms scaled_target_total
#print axioms scaled_draft_total
#print axioms accepted_residual_total
#print axioms accepted_rejected_total
#print axioms residual_equals_rejection
#print axioms output_mass_closed_form
#print axioms output_mass_target_identity
#print axioms output_mass_total
#print axioms output_space_positive
#print axioms exact_target_distribution

end SpeculativeSampling
