# Frontier tensor empirical mathematics

## Literature review and research program for MAX/Mojo

**Status:** literature review and research-program baseline  
**Current through:** 23 September 2026  
**Repository role:** non-authoritative research guidance for `llm-kernel-lab`

## Executive summary

This review surveys the literature most relevant to a tensor empirical-mathematics
program built around MAX/Mojo. The goal is not to identify fields that merely
"use tensors"; it is to identify research frontiers where optimized tensor
kernels can materially enlarge the empirical regime and where the resulting
observations can be converted into reproducible, independently checked
mathematical evidence.

Three programs stand out.

| Program | Scientific opportunity | Why larger computation matters | Main kernel opportunity | Main risk |
| --- | --- | --- | --- | --- |
| **Moment polytopes / tensor scaling** | Discover new geometric and invariant-theoretic structure, candidate facets, and orbit-closure phenomena | Recent computations have already led to structural results and reach beyond previously charted formats | `mode_gram`, mode transforms, contractions, batched eigensystems/rank tests | Numerical facets and rank drops are not proofs |
| **Spiked tensor PCA / phase transitions** | Resolve finite-size computational-statistical transitions and compare algorithm classes under common normalization | Theory predicts multiple algorithm-dependent regimes and smooth runtime/SNR tradeoffs that are only partly visible asymptotically | high-order contraction, implicit noise generation, matrix-free spectral operators, massively parallel trials | Threshold normalizations differ across papers; generic hardness remains conjectural |
| **Tensor-network turbulence and kinetic PDEs** | Measure when high-dimensional evolution remains compressible and identify rank-growth laws or crossovers | Recent work shows strong compression, but no universal rank-growth law in Reynolds number, intermittency, stiffness, or time is established | `tt_round`, `tt_apply`, fused PDE updates, small batched QR/SVD, contractions | Rank is strongly tensorization-, tolerance-, and observable-dependent |

The strongest theorem-generation opportunity is the first program. Tensor
scaling was placed on polynomial-time foundations, then incorporated into a
broader noncommutative/geodesic optimization framework. Recent work on moment
polytope computation has pushed complete or high-probability computation into
new tensor formats and has directly motivated structural discoveries about
matrix-multiplication tensors. That makes it unusually plausible that a faster
empirical instrument can expose objects that later admit exact proof.

The spiked-tensor program is the cleanest controlled empirical laboratory.
Kikuchi/Sum-of-Squares-style methods, communication-complexity lower bounds,
weighted-hypergraph methods, local MCMC, and newer local-search algorithms all
show that the landscape is richer than a single "easy/hard" threshold. Large,
carefully normalized Monte Carlo campaigns can test finite-size scaling,
universality, and algorithm-specific phase transitions.

The tensor-network PDE program has the highest scientific-computing upside.
Recent turbulence and kinetic-equation papers show that tensor representations
can be dramatically more compact than full grids in favorable regimes, but the
literature does not yet establish a universal law for the rank required to
maintain a given accuracy. The core empirical object should therefore be a
rank surface such as

[
r_arepsilon(mathrm{Re},N,t,mathcal T,Q),
]

where (mathcal T) records tensorization and (Q) records the accuracy
criterion or physical observable.

MAX/Mojo is best used as the instrument, not the authority. The program should
separate:

[
	ext{high-throughput search}
longrightarrow
	ext{high-precision/statistical replication}
longrightarrow
	ext{exact or domain-specific validation}.
]

For pure tensor geometry this means rational/algebraic reconstruction and exact
rank or membership checking. For statistical inference this means independent
regeneration, confidence intervals, and finite-size scaling. For PDEs it means
field error together with physical observables, conservation checks, spectra,
structure functions, dissipation, and tail statistics.

---

## 1. Scope and evidence standards

The review emphasizes primary and high-impact work from 2018 through September
2026, with emphasis on 2022--2026. Earlier papers are included where they
establish the algorithmic or geometric framework needed to interpret recent
results.

The repository authority ladder remains unchanged:

1. profiling and timing are engineering evidence;
2. floating-point observations are candidate phenomena;
3. replicated numerical observations are stronger empirical evidence;
4. exact reconstructed witnesses are algebraic evidence;
5. independently verified certificates or Lean-checked theorems are the only
   theorem-grade artifacts.

A GPU result is never promoted solely because it is stable or fast.

Tensor-PCA papers frequently use different signal/noise normalizations. Raw
values of (lambda) must therefore not be compared across papers without an
explicit conversion to a common dimensionless SNR.

Likewise, "rank" in tensor-network PDE papers may mean a TT/MPS bond dimension
under a particular tensorization and truncation tolerance. It is not meaningful
without recording the representation, ordering, norm, tolerance, and observable.

---

## 2. Moment polytopes, tensor scaling, and computational invariant theory

### 2.1 State of the field

Tensor scaling asks whether local invertible transformations can move a tensor
to prescribed reduced marginals. The problem links invariant theory,
representation theory, quantum marginal problems, orbit closure, null-cone
membership, and algebraic complexity.

Bürgisser, Franks, Garg, Oliveira, Walter, and Wigderson developed efficient
algorithms for tensor scaling and weak membership in moment polytopes. This
gave a polynomial-time algorithmic foundation for treating marginal data as
something that can be explored computationally rather than only symbolically.

Allen-Zhu, Garg, Li, Oliveira, and Wigderson developed operator scaling through
geodesically convex optimization. Bürgisser and collaborators then generalized
this into a broader framework for noncommutative optimization, moment maps,
null cones, and moment polytopes.

Recent work has moved from "efficiently optimize within known structure" to
"compute explicit new structure." In 2025, van den Berg and collaborators
reported a general computational method for moment polytopes and computations
covering all (3	imes3	imes3) tensor moment polytopes and high-probability
results in (4	imes4	imes4), substantially extending the previously tractable
regime. Separate 2025 work proved that the moment polytope of matrix
multiplication is not maximal and linked this phenomenon to matrix-subspace
minrank and border subrank.

This is the clearest precedent in the review for a pipeline of the form:

[
	ext{large computation}
ightarrow
	ext{unexpected geometric pattern}
ightarrow
	ext{exact separating structure}
ightarrow
	ext{theorem}.
]

### 2.2 Key sources

- Bürgisser et al., **Efficient algorithms for tensor scaling, quantum
  marginals and moment polytopes** (2018):  
  <https://arxiv.org/abs/1804.04739>
- Allen-Zhu et al., **Operator Scaling via Geodesically Convex Optimization,
  Invariant Theory and Polynomial Identity Testing** (2018):  
  <https://arxiv.org/abs/1804.01076>
- Bürgisser et al., **Towards a theory of non-commutative optimization**
  (2019):  
  <https://arxiv.org/abs/1910.12375>
- Vergne and Walter, **Moment cone membership for quivers in strongly
  polynomial time** (2023):  
  <https://arxiv.org/abs/2303.14821>
- van den Berg et al., **The moment polytope of matrix multiplication is not
  maximal** (2025):  
  <https://arxiv.org/abs/2503.22633>
- van den Berg et al., **Computing moment polytopes—with a focus on tensors,
  entanglement and matrix multiplication** (2025):  
  <https://arxiv.org/abs/2510.08336>

### 2.3 Research gaps

Several gaps are attractive for empirical work.

First, complete explicit knowledge remains sparse outside small formats. Second,
matrix-multiplication non-maximality raises quantitative questions: how do
separating inequalities behave with dimension, and are there scalable families
of witnesses rather than isolated low-dimensional accidents? Third, the quiver
literature leaves neighboring strongly polynomial semi-invariant questions
open.

The following should be treated as hypotheses to test, not as literature facts:

- structured orbit families may exhibit recurring low-complexity facet normals;
- tensor-scaling trajectories may contain enough information to predict useful
  exact separating inequalities;
- matrix-multiplication and iterated-multiplication families may admit scalable
  families of non-maximality witnesses.

### 2.4 MAX/Mojo opportunity

The central dense primitive is

[
G_i=T_{(i)}T_{(i)}^ast.
]

For dense (d^3) tensors this is roughly (2d^4) operations per mode. The
systems challenge is not only the matrix multiply; it is avoiding physical
mode permutations, fusing marginal normalization, exploiting symmetry, and
efficiently batching medium-sized eigensystems and rank tests.

The first kernel family should therefore include:

- `mode_gram`;
- layout-only mode transforms when possible;
- fused marginal normalization;
- `batched_rank`;
- sparse-to-dense transforms for structured tensors such as matrix
  multiplication.

MAX/Mojo's layout machinery is directly relevant because fixed tile structure
can coexist with runtime tensor dimensions:
<https://docs.modular.com/mojo/layout/tile_layout/>.

A useful research sequence is:

[
	ext{Mojo numerical anomaly}
ightarrow
	ext{FP64/high-precision replay}
ightarrow
	ext{rational/algebraic reconstruction}
ightarrow
	ext{exact Sage/Julia checker}.
]

The GPU is a candidate generator, never the rank or polytope authority.

---

## 3. Spiked tensor PCA and computational-statistical phase transitions

### 3.1 State of the field

Tensor PCA studies a planted low-rank signal hidden in high-order noise, for
example

[
Y=lambda x^{otimes p}+W.
]

Unlike matrix PCA, the tensor problem exhibits a gap between what is
statistically identifiable and what known efficient algorithms achieve.

The modern picture is not a single threshold. Wein, El Alaoui, and Moore's
Kikuchi hierarchy gives a family of spectral/message-passing-inspired methods
whose computational cost and statistical power vary together. Dudeja and Hsu
provide lower bounds for restricted computational models using communication
complexity. Zhou, Basso, and Mei compare fixed-depth QAOA to classical tensor
power behavior. Chen, Sheehan, and Zadik identify sharp low-temperature local
MCMC thresholds in sparse tensor PCA. Lovig, Sheehan, Tsirkas, and Zadik show
that carefully designed local search can operate in stronger regimes. Li's
2025 weighted-hypergraph result gives an explicit smooth runtime/SNR tradeoff
for tensor PCA.

Together these results motivate an empirical surface rather than a single
number:

[
P_{mathrm{recover}}(n,p,mathrm{SNR},A,	ext{budget}).
]

### 3.2 Key sources

- Wein, El Alaoui, and Moore, **The Kikuchi Hierarchy and Tensor PCA**
  (FOCS 2019):  
  <https://arxiv.org/abs/1904.03858>
- Dudeja and Hsu, **Statistical-Computational Trade-offs in Tensor PCA and
  Related Problems via Communication Complexity** (2022):  
  <https://arxiv.org/abs/2204.07526>
- Zhou, Basso, and Mei, **Statistical Estimation in the Spiked Tensor Model
  via QAOA** (2024):  
  <https://arxiv.org/abs/2402.19456>
- Chen, Sheehan, and Zadik, **On the Low-Temperature MCMC threshold**
  (2024):  
  <https://arxiv.org/abs/2408.00746>
- Lovig, Sheehan, Tsirkas, and Zadik, **Almost-Optimal Local-Search Methods
  for Sparse Tensor PCA** (2025):  
  <https://arxiv.org/abs/2506.09959>
- Li, **A Smooth Computational Transition in Tensor PCA** (2025):  
  <https://arxiv.org/abs/2509.09904>
- Tabanelli et al., **Computational Thresholds in Multi-Modal Learning via
  the Spiked Matrix-Tensor Model** (2025):  
  <https://arxiv.org/abs/2506.02664>

### 3.3 Research gaps

A strong finite-size program should test at least four questions.

**Finite-size scaling.** For an algorithm (A), estimate whether

[
widehat{lambda}_c(n,A)
=
lambda_c(A)+c_A n^{-omega_A}+o(n^{-omega_A})
]

describes the crossover, and whether exponents differ by algorithm class.

**Universality.** Test whether Gaussian, Rademacher, and mild non-Gaussian
ensembles share the same rescaled finite-size crossover.

**Equal-work comparison.** Compare power methods, unfolding, selected Kikuchi
levels, weighted-hypergraph-inspired statistics, MCMC, and local search under
both equal wall-clock time and a hardware-independent work count.

**Multi-modal staircase transitions.** Test whether the apparent staged
recovery behavior in coupled matrix/tensor models persists under finite-size
scaling.

These experiments can disprove naive universal pictures even when they cannot
establish generic computational hardness.

### 3.4 MAX/Mojo opportunity

A direct order-three tensor-vector contraction is approximately memory-bandwidth
bound:

[
y_i=sum_{jk} T_{ijk}x_jx_k.
]

For explicit FP32 tensors, storage becomes the limiting factor rapidly. The
better kernel is therefore not simply "faster contraction" but deterministic
implicit-noise contraction:

`fused_rng_contract(seed, block_id, x, signal) -> y`

A counter-based generator can reproduce each noise tile, contract immediately,
and discard it. This converts a memory-capacity problem into a streaming/RNG
problem and permits experiments beyond single-GPU HBM capacity.

A second target is a matrix-free `kikuchi_matvec` that generates the lifted
operator's local combinatorial structure on demand rather than materializing a
large matrix.

Sparse tensor PCA suggests a third family:

- `local_energy_delta`;
- `random_threshold_accept`;
- `topk_swap`;
- `multi_chain_reduce`.

The experiment harness must record seeds, distribution conventions, work
counts, dtype, accumulation mode, and normalization. Low precision must be
validated distributionally because numerical changes to the effective noise
law can move an apparent transition.

---

## 4. Tensor networks for turbulence and kinetic PDEs

### 4.1 State of the field

Tensor trains and matrix-product-state representations replace a full
high-dimensional array with a chain of low-order cores. In favorable regimes
the storage can change from an exponential grid to roughly

[
O(DNr^2),
]

or even smaller under quantized tensorization, provided the required rank (r)
remains moderate.

Recent turbulence papers have moved tensor networks from demonstrations on
smooth or synthetic fields into developed turbulent flow and turbulence
probability distributions.

Gourianov, Givi, Jaksch, and Pope showed that tensor networks can represent and
evolve high-dimensional turbulence probability distributions. Pisoni and
collaborators studied compression, simulation, and synthesis of turbulent
flows using tensor trains and reported favorable scaling in the regimes they
tested. Esmaeili and collaborators studied tensor-network encoding of
isotropic turbulence and passive scalars and emphasized that small-scale
observables such as dissipation can degrade earlier than large-scale energy
statistics. Pisoni, Tiunov, and Calascibetta developed a hybrid TT approach for
intermittent passive-scalar turbulence.

For kinetic equations, Wang and Hu developed dynamical tensor-train methods for
kinetic equations and later extended the framework to stiff Fokker--Planck
collisions. These methods attack a setting where full phase-space
discretization is prohibitively expensive.

### 4.2 Key sources

- Gourianov et al., **Tensor networks enable the calculation of turbulence
  probability distributions** (*Science Advances*, 2025):  
  <https://www.science.org/doi/10.1126/sciadv.ads5990>  
  Preprint: <https://arxiv.org/abs/2407.09169>
- Pisoni et al., **Compression, simulation, and synthesis of turbulent flows
  with tensor trains** (2025/2026):  
  <https://arxiv.org/abs/2506.05477>
- Esmaeili et al., **A priori Assessment of Tensor-Network Encoding for
  Isotropic Turbulent Flows** (2026):  
  <https://arxiv.org/abs/2608.28869>
- Pisoni, Tiunov, and Calascibetta, **Multiscale passive scalar turbulence in
  a compressed subspace via tensor trains** (2026):  
  <https://arxiv.org/abs/2608.00194>
- Pinkston et al., **Matrix Product State Simulation of Reacting Shear
  Flows** (2025):  
  <https://arxiv.org/abs/2512.13661>
- Wang and Hu, **Dynamical Tensor Train Approximation for Kinetic Equations**
  (2025; JCP 2026):  
  <https://arxiv.org/abs/2512.14950>
- Wang and Hu, **Implicit Dynamical Tensor Train Approximation for Kinetic
  Equations with Stiff Fokker–Planck Collisions** (2026):  
  <https://arxiv.org/abs/2605.15382>

### 4.3 Research gaps

The literature does not establish a universal law of the form

[
rpropto mathrm{Re}^{alpha}.
]

A useful empirical object is instead

[
r_arepsilon=
r_arepsilon(
mathrm{Re},
mathrm{Pe},
N,
t,
mathcal T,
Q).
]

The experiment should distinguish at least three possibilities:

1. **mild resolution growth:** rank or parameter count grows slowly with mesh
   refinement at fixed physics;
2. **physical-complexity growth:** rank follows a reproducible law in Reynolds,
   Péclet, stiffness, or distance from equilibrium over an intermediate regime;
3. **crossover behavior:** intermittency, nonlinear products, reactions, or
   stiff collisions induce rapid rank growth beyond some parameter range.

These are competing empirical hypotheses. None should be assumed.

### 4.4 MAX/Mojo opportunity

The main tensor-train primitive is `tt_round`. A conventional rounding sweep
uses QR orthogonalization and truncated SVD. Under uniform physical dimension
(q), order (D), and rank (r),

[
	ext{storage}sim O(Dqr^2),
qquad
	ext{rounding}sim O(Dqr^3).
]

The practical problem is often intermediate rank (R), not only the retained
rank (r). Nonlinear products or operator application can temporarily produce
large (R), making subsequent factorization expensive. Avoiding rank explosion
may therefore matter more than accelerating a standalone SVD.

The kernel family should include:

- `tt_round`;
- `tt_apply`;
- `tt_axpy`;
- `tt_inner`;
- small batched QR/SVD;
- fused advection/core updates;
- fused collision/core updates;
- rank telemetry at every step.

The turbulence program should begin with static snapshot compression, then
progress through linear transport, passive scalar evolution, nonlinear
Navier--Stokes, and reacting flow. Every result should record both algebraic
error and physically meaningful diagnostics.

The kinetic program should similarly record rank versus time, stiffness,
collision rate, forcing, and distance from equilibrium.

---

## 5. MAX/Mojo systems layer

The MAX/Mojo stack is relevant because the useful optimization targets lie
around GEMM rather than inside generic GEMM alone: permutations, contraction
fusion, reductions, marginal construction, small/medium factorizations,
implicit random generation, and structured sparse+dense operators.

Current Modular documentation provides:

- Mojo language and GPU programming:
  <https://docs.modular.com/mojo/>
- Mojo hardware/requirements:
  <https://docs.modular.com/mojo/requirements/>
- Mojo vision and MAX integration:
  <https://docs.modular.com/mojo/vision/>
- layout and tiling:
  <https://docs.modular.com/mojo/layout/tile_layout/>
- experimental PyTorch custom-operation integration:
  <https://docs.modular.com/max/api/python/experimental.torch/>

The PyTorch bridge should be treated as an integration surface, not as part of
the scientific contract. Experiment descriptions and result schemas should be
runtime-independent so that a reference implementation can replay the same
case.

The optimization order should be:

1. eliminate unnecessary copies;
2. increase data reuse;
3. batch small operations;
4. fuse producer/consumer kernels;
5. only then optimize tensor-core mapping.

Vendor BLAS, vendor eigensolvers/SVD, PyTorch, and CUTLASS-style baselines must
remain comparison points. A custom Mojo kernel that fails to outperform a
mature baseline should be retained as a negative result rather than hidden.

### Precision policy

| Tier | Use |
| --- | --- |
| BF16/FP16/TF32 | candidate generation only where validated |
| FP32 | default empirical sweeps |
| FP64 | boundary cases and numerical reference |
| arbitrary/exact | authoritative algebraic reconstruction |

For tensor PCA, distributional fidelity must be checked under any reduced
precision. For moment-polytopes, small spectral gaps near boundaries make
low-precision facet inference particularly risky.

---

## 6. Reproducibility contract

Every experimental artifact should record at least:

```text
experiment_id
problem_contract_hash
kernel_revision
MAX/Mojo revision
reference_implementation_revision
device and topology
dtype and accumulation dtype
deterministic-reduction policy
seed and random-stream construction
shape / tensorization / sparsity
input or dataset hash
algorithm-specific normalization
raw observation
performance counters
reconstruction or statistical-validation status
checker / validation version
authority grade
```

Frontier-specific additions:

- **Moment geometry:** exact candidate inequality or rank witness, rational
  reconstruction, exact residual.
- **Tensor PCA:** common SNR convention, independent seeds, confidence interval,
  work count, recovery metric.
- **PDE/TN:** tensorization, truncation tolerance, maximum/intermediate ranks,
  spectra/structure functions/dissipation/conservation metrics.

---

## 7. Concrete research program

### Phase 1: common kernels and reproductions

Implement and validate:

```text
contract
mode_gram
batched_rank
tt_round
fused_rng_contract
```

Reproduce one published result in each frontier before attempting novelty.

For tensor geometry, reproduce small-format tensor-scaling/marginal behavior and
at least one published moment-polytope case.

For tensor PCA, reproduce baseline power/unfolding transition curves under a
canonical normalization and verify the random generator statistically.

For tensor dynamics, reproduce tensor-train compression on published or public
turbulence/kinetic benchmarks and validate physical observables, not only
Frobenius error.

### Phase 2: new empirical surfaces

**Moment geometry**
- structured matrix-multiplication scans;
- random orbit-family scans;
- candidate low-complexity facet reconstruction;
- exact certification of every published witness.

**Tensor PCA**
- (p=3,4) finite-size atlas;
- implicit-noise experiments beyond single-GPU storage;
- selected Kikuchi/hypergraph operators;
- sparse local-search/MCMC comparison;
- universality tests across noise ensembles.

**Tensor dynamics**
- rank/error surfaces across resolution and physical parameters;
- passive-scalar intermittency;
- stiff kinetic collision benchmarks;
- measurement of intermediate-rank explosions.

### Phase 3: promotion to theorem or durable scientific claim

A candidate result can be promoted only when the validation matches its domain:

- exact certificate for algebraic geometry;
- statistical replication and uncertainty quantification for phase transitions;
- independent implementation plus physical diagnostics for PDE rank laws.

---

## 8. Comparative assessment

The three programs are complementary.

**Moment polytopes / tensor scaling** has the shortest credible path from
accelerated search to exact mathematical discovery because the recent
literature already demonstrates that explicit computation can expose new
structure.

**Spiked tensor PCA** has the cleanest experiment design and strongest control
over data generation, making it the best environment for building and auditing
the empirical methodology.

**Tensor-network turbulence and kinetic PDEs** carries the largest systems and
scientific-computing upside, but should begin as a disciplined measurement of
rank growth rather than as an attempt to replace mature CFD or kinetic solvers.

A sensible initial allocation is therefore to put the largest share of
mathematical-search effort into moment geometry, use tensor PCA as the
controlled statistical benchmark, and treat tensor dynamics as a rank-law
measurement program until the data justify deeper solver investment.

---

## 9. Immediate repository implications

This review reinforces the existing `RESEARCH_PLAN.md` rather than replacing
it.

The first reusable kernel vocabulary remains deliberately small:

```text
contract
mode_gram
batched_rank
tt_round
fused_rng_contract
```

The next repository-level work should be:

1. add one transparent CPU/PyTorch reference for each primitive;
2. add Mojo implementations behind a narrow interface;
3. create property/differential tests on bounded cases;
4. define frontier-specific experiment manifests;
5. add exact/statistical/domain-validation adapters;
6. preserve negative performance and negative scientific results.

The research objective is not "prove Mojo is faster." It is:

> use MAX/Mojo to make a previously inaccessible empirical regime observable,
> then convert any interesting observation into a reproducible artifact whose
> authority comes from an exact or independently validated boundary.
