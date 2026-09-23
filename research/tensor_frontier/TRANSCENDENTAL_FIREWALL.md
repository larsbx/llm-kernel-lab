# Transcendental firewall

## Purpose

Tensor literature often introduces geometric or probabilistic structure through
angles, trigonometric coordinates, phases, exponentials, logarithms, or
square-root normalizations.  Those are frequently convenient coordinates rather
than the invariant itself.

This lab rewrites the *mathematical contract* algebraically wherever that can be
done without changing the claim.

## Canonical algebraic vocabulary

Let a symmetric bilinear form be `B` and `Q(x)=B(x,x)`.

### Angle-free pair geometry

For non-isotropic `x,y`, use the Gram data

```text
Qx = Q(x)
Qy = Q(y)
Bxy = B(x,y)
Delta = Qx*Qy - Bxy^2
spread = Delta / (Qx*Qy)
```

No angle is reconstructed.  Scale invariance is immediate under independent
nonzero rescaling of `x` and `y`.

### Trigonometric pairs -> algebraic rotors

Replace a sine/cosine pair by a conic point `(c,s)` satisfying

```text
c^2 + s^2 = 1.
```

A rational parameter `t`, when its denominator is invertible, gives

```text
c = (1 - t^2)/(1 + t^2)
s = 2t/(1 + t^2).
```

Composition is polynomial:

```text
(c1,s1) o (c2,s2)
  = (c1*c2 - s1*s2, s1*c2 + c1*s2).
```

The corresponding operator is the 2x2 rotor matrix

```text
[[ c, -s],
 [ s,  c]].
```

There is no angle object in the contract.

### Tangent -> projective ratio

Use the projective pair `[s:c]`, not `s/c`, so the representation remains
defined when `c=0`.  Affine division is performed only when the denominator is
known nonzero.

### Complex phase -> quadratic algebra element

A phase-like object is represented as a pair `(a,b)` in a quadratic algebra
with multiplication fixed by its defining polynomial.  A norm-one condition is
a polynomial equation.  The notation `exp(i*theta)` is not part of an
authoritative contract.

### Norm -> quadrance

Prefer `Q(x)` or squared residuals to square-root norms.  Square root is
algebraic rather than transcendental, but the authority layer remains
radical-free unless an explicit algebraic extension is part of the problem.

## Audit table

| Source vocabulary | Algebraic treatment | Class |
| --- | --- | --- |
| angle | Gram data / spread | A |
| sin, cos pair | conic rotor `c^2+s^2=1` | A |
| tan | projective ratio `[s:c]` | A |
| inverse trig | retain invariant; do not recover angle | A when only comparison/invariant is needed |
| unit complex phase | norm-one quadratic-algebra pair / rotor matrix | A |
| finite Fourier phase | algebraic root relation / recurrence | A |
| softmax probabilities | start from nonzero weights and normalize algebraically | A for the probabilities, not for logits |
| Gaussian model | finite generator + moment/covariance contract | A for sampled experiment contract |
| sqrt norm | quadrance / squared residual | A when only norm comparison/residual is needed |
| Shannon entropy | power sums / collision statistics | S |
| log determinant | determinant/resultant/valuation when that is the actual predicate | A or S, case-by-case |
| general exp/log parameterization | multiplicative variable or rational chart when semantics permit | A or S |
| general matrix exponential | rational/Cayley chart only where group semantics permit | S otherwise |
| analytic asymptotic free energy | retain as analytic diagnostic | D |

A = exact algebraization. S = non-equivalent surrogate. D = diagnostic only.

## Non-equivalence rule

The firewall is not permission to rename a different quantity.

Examples:

- a Renyi/power-sum statistic is not Shannon entropy;
- a Cayley transform is not the general matrix exponential;
- squared overlap can replace an angle-based success predicate only when the
  predicate depends solely on that overlap;
- a sampled finite noise generator is not automatically equivalent to every
  theorem stated for an ideal continuous distribution.

Whenever equivalence is not proved, the experiment records S or D.

## Source-code gate

`tools/audit_transcendentals.py` rejects direct use of common trig,
inverse-trig, exp/log, hyperbolic, square-root, `pi`, and complex-constructor
syntax inside the authoritative `tensor_lab/` source layer.  It is deliberately
narrow: documentation may discuss forbidden vocabulary, and numerical benchmark
adapters may eventually use analytic routines if their output remains E0-E2.

The source-code gate complements, rather than replaces, the semantic audit above.
