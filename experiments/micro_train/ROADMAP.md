# micro-train roadmap

Milestones for the micro LM / text-classifier pipeline, in dependency order.
Each milestone has one goal, the deliverables that reach it, **exit criteria a
reviewer can check** and the **evidence** it must leave in the repository. A
milestone is done when its exit criteria hold on `main`, not when its code merges.

Scope stays fixed throughout: byte-level models trained from scratch, small enough
for one CPU or one GPU, provisioned by `flake.nix`. This is the **experiments**
plane of `estate.toml`; nothing here is proof evidence, and the promotion and
deletion criteria in the [README](README.md#promotion-and-deletion) govern its end.

## Where it stands: M0 (done, `086c70c`, moved in #9)

- `micro-train lm` (causal transformer, held-out bits per byte) and
  `micro-train classify` (bidirectional encoder over JSONL `text`/`label`, eval
  accuracy); seeded; writes `model.safetensors`, `config.json`, `metrics.json`.
- 8 tests, run inside every Nix build of the package.
- `nix run .#run` loads a content-tagged OCI image into podman and trains in it;
  verified on CPU (toy LM 1.74 bits/byte, toy classifier 96% eval accuracy).

Known gaps, each owned by a milestone below:

| Gap | Owner |
| --- | --- |
| a saved model cannot be loaded back; no `predict` | M1 |
| no `flake.lock`; metrics record no provenance; no CI for the flake | M2 |
| the classifier is scored on one held-out file that also steers tuning; nothing checks split overlap or repeated test reads; no baseline, no uncertainty | M3 |
| only synthetic toy data | M4 |
| evaluation loads every row into memory (`load_labelled` reads the whole file) and runs the eval set as one batch; no validation-based stopping, schedule, or resume | M5 |
| CUDA image evaluates but has never been built or run | M7 |

## M1: Load and predict

**Goal.** A trained model is a usable artifact, not a write-only file.

- `micro_train.load(run_dir)` rebuilds the model from `config.json` +
  `model.safetensors` and refuses a mismatched or unknown `mode`.
- `micro-train predict --model runs/clf --input rows.jsonl` emits one JSONL row per
  input (`label`, per-class probabilities); `micro-train sample` for LMs.
- Evaluation code calls the same loader, so what is measured is what is shipped.

**Exit criteria.** Round-trip test: train → save → load → identical logits on
fixed inputs. `predict` on the training run's eval file reproduces
`metrics.json`'s `eval_accuracy` exactly. A run directory from another mode or a
corrupted weight file is rejected, not silently loaded.

**Evidence.** Tests under `tests/`; the `predict` path exercised in the podman
smoke run.

## M2: Reproducible runs

**Goal.** Any number in this repository can be regenerated from what it records.

- Commit `flake.lock` (first `nix` command on a machine that can reach
  `github:NixOS/nixpkgs`).
- `metrics.json` gains a `provenance` block: git revision, image tag, torch
  version, device, SHA-256 of every input file, full CLI arguments.
- CI job: `nix flake check` (package build + tests) and one podman smoke run per
  mode on the CPU image.

**Exit criteria.** Two CI runs of the same commit and seed produce byte-identical
`metrics.json` (provenance aside from timestamps). A run whose recorded input hash
does not match its data fails `predict --verify`.

**Evidence.** CI workflow; a committed example run under `results/micro/`.

## M3: Honest classifier evaluation

**Goal.** A reported test score is the first and only recorded read of a split
that is disjoint from everything the model was selected on.

A separate command alone guarantees neither: `--val` and `--test` can name the same
data, and `evaluate` can be rerun while hyperparameters are still moving. So the
protocol is enforced, not documented:

- **Split disjointness.** Every command that reads more than one split refuses if
  two of them resolve to the same file or the same content (SHA-256), or if any
  row's text, after Unicode NFC and whitespace normalisation, occurs in more than
  one of train, val and test. Near-duplicates are out of scope and say so in the
  report.
- **Selection never sees test.** `train` and `classify` accept `--train` and
  `--val` only; early stopping and checkpoint choice use `--val`.
- **Freeze, then read once.** `micro-train freeze <run>` records the selected
  model's weight hash, config, val metrics, and the normalised row hashes of its
  train and val splits. `micro-train evaluate --final --test FILE <run>` refuses
  an unfrozen run, checks each test row against those hashes as it streams past
  (so the test-side check retains nothing), then appends
  `{test_sha256, model_sha256, frozen_at, evaluated_at}` to
  `results/micro/<dataset>/test-ledger.jsonl` and refuses if that test hash already
  has an entry. A spent test split is replaced, not reread.
- **The ledger is append-only in git.** A CI check compares the ledger with the
  base branch and fails on any change other than appended lines, so a second read
  cannot be erased to make room for a better one.
- **Stated limit.** Nothing in-repo can detect the test file being read outside
  the tool. The claim is therefore exactly the goal above: first and only
  *recorded* read, of a model frozen before it.
- Metrics: accuracy, macro-F1, per-class precision/recall, confusion matrix,
  expected calibration error; 95% bootstrap intervals on accuracy and macro-F1.
- Baselines on the same splits: majority class, and byte n-gram logistic
  regression.

**Exit criteria.** Tests show each refusal: `--val` and `--test` naming the same file;
identical content under two paths; one shared row across train and test after
normalisation; `--final` on an unfrozen run; a second `--final` on the same test
hash; and the CI ledger check failing on an edited or deleted line. On a fixed
dataset the report shows the model and both baselines with intervals, the ledger
entry it came from, and plainly whether the model's interval clears the stronger
baseline's. A label present in test but not train fails closed (as eval does
today).

**Evidence.** `results/micro/<dataset>/report.json` and a rendered table.

## M4: Real datasets

**Goal.** Results on public data, not toys.

- `micro-train data fetch <name>` for two or three small public text
  classification sets (a topic set and a sentiment set at minimum), each pinned by
  URL and SHA-256 and converted to the JSONL format; a short data card per set
  (source, license, label meanings, split sizes).
- Fetching is a separate step: training stays offline.

**Exit criteria.** `fetch` refuses a download whose hash differs from the pin. Each
dataset has an M3 report.

**Evidence.** `results/micro/` reports; data cards next to the fetch definitions.

## M5: Training that scales past toys

**Goal.** Evaluation runs in memory bounded independently of eval-set size, on the
host as well as the accelerator; training stays correct on M4-sized data.

Batching inference alone does not bound host memory: `load_labelled` reads the
whole file and keeps every row, text and label before evaluation starts. So:

- **Streaming input.** Evaluation reads JSONL line by line into fixed-size batches;
  nothing holds the full split. The unseen-label check runs on the fly against the
  training label set (labels, not rows).
- **Streaming metrics.** Accuracy, per-class counts, the confusion matrix and
  calibration bins are running sums; bootstrap intervals use a streaming (Poisson)
  bootstrap with a fixed number of replicate counters instead of per-example
  storage.
- **Batched inference** at a fixed batch size (today the whole eval set is one
  tensor).
- **Scoped exception.** Training still loads its split, since it samples rows at
  random; that is a stated bound on training, not on evaluation. M3's
  disjointness check keeps one 32-byte hash per train and val row, likewise
  linear and stated.
- Warmup + cosine schedule, gradient clipping, class weighting option.
- Early stopping and best-checkpoint selection on `--val`; checkpoint and
  `--resume`.

**Exit criteria.** Evaluation on two eval sets differing in size by at least 10×
shows the same peak host RSS and peak accelerator memory within a stated bound,
with the reader and metric accumulators also checked under `tracemalloc`. Streaming
metrics equal the in-memory computation on a set small enough for both, and the
streaming bootstrap's intervals agree with the exact bootstrap within a stated
tolerance. Resuming from a checkpoint reproduces an uninterrupted run's final
metrics for the same seed.

**Evidence.** Tests for batching and resume; an M3 report showing the effect of
the schedule against the M4 baseline run.

## M6: Does LM pretraining help the classifier?

**Goal.** Answer one question with a controlled experiment: does pretraining the
byte transformer as an LM on unlabeled text improve classification at small label
budgets?

- `micro-train classify --init runs/lm` initialises the encoder from an LM run
  (same width/depth), fine-tuning with the existing head.
- Sweep labelled-set size (e.g. 1%, 10%, 100%) × {scratch, pretrained}, 3 seeds
  each.

**Exit criteria.** A results table with intervals over seeds and a stated verdict
(helps / no measurable effect / hurts) per budget. A null result is an acceptable
outcome and is recorded as one.

**Evidence.** `results/micro/pretraining/` with the sweep's reports.

## M7: GPU path verified

**Goal.** The `.#run-cuda` variant is as trustworthy as the CPU one.

- Build and run `image-cuda` on a CDI host (e.g. a Runpod RTX 4090, driver ≥ 580).
- Record throughput (examples/s, CPU vs GPU) and the GPU determinism caveats
  (which metrics are bit-stable across runs, which are not).

**Exit criteria.** M2's smoke runs pass on the GPU image with `device: "cuda"` in
`metrics.json`; CPU and GPU agree on metrics within a stated tolerance.

**Evidence.** A run record under `results/micro/gpu/`; the README's GPU section
updated from "untested" to what was measured.

## M8: Promote or delete

**Goal.** Close the experiment, per the README's criteria.

- **Promote** if a second consumer trains with it: freeze the CLI and the
  `metrics.json`/`report.json` schemas as a versioned contract, then move it to its
  own plane or a shared estate package (and its own repository if the consumer is
  outside `llm-kernel-lab`).
- **Delete** if no consumer appears and no run has been recorded within the
  README's window.

**Exit criteria.** One of the two happened, recorded in `estate.toml` and in a
decision note under `docs/`.

## Order and parallelism

```text
M0 ─▶ M1 ─▶ M2 ─▶ M3 ─▶ M4 ─▶ M5 ─▶ M6 ─▶ M8
              │                             ▲
              └──────────▶ M7 ──────────────┘
```

M1 and M2 are small and unblock everything. M3 must precede any reported real-data
number, and its ledger is what makes such a number reportable. M7 needs only M2 and can run whenever a GPU is available.
