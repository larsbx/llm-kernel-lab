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
| the classifier is scored on one held-out file that also steers tuning; no split-overlap check, frozen-artifact verification, or authoritative test reservation; no baseline, no uncertainty | M3 |
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

**Goal.** A reported test score comes from the first and only recorded evaluation
attempt on a split disjoint from everything the models were selected on. That
attempt evaluates one frozen group: the selected model, both baselines, and any
other comparisons declared before test access.

M3 is pending. A separate command and a branch-local ledger do not enforce this
goal. Its implementation must enforce the following protocol:

- **Split disjointness.** Every command that reads more than one split refuses if
  two of them resolve to the same file or the same content (SHA-256), or if any
  row's text, after Unicode NFC and whitespace normalisation, occurs in more than
  one of train, val and test. Near-duplicates are out of scope and say so in the
  report. Test-side content and row checks run only after the reservation below;
  the check itself spends the attempt even if it finds overlap.
- **Selection never sees test.** `train` and `classify` accept `--train` and
  `--val` only; early stopping and checkpoint choice use `--val`.
- **Trusted split identity.** Before model selection, register a dataset manifest
  with the evaluation authority: dataset/version, the SHA-256 of the exact test
  JSONL bytes, and an immutable object containing those bytes. The evaluator gets
  the expected hash from that manifest, never from a caller-supplied hash or by
  scanning the selected test file before reservation. `evaluate --final` takes
  `--dataset NAME` and resolves its immutable input through the registered
  manifest. The evaluator cannot replace the manifest or repin a spent split.
  Paths, dataset aliases, run IDs and branches
  are not reservation keys; the same test-byte hash has one key across the project.
- **Immutable freeze receipt.** `micro-train freeze <run>` registers a receipt
  with the authority before any test access. It binds every model and baseline
  in the group to its weight/artifact SHA-256, effective selection and evaluation
  configuration, validation metrics, and train/val file and normalised row hashes.
  Include label order, tokenizer/preprocessing, context/truncation, checkpoint
  choice, calibration/decision rules, metric/bootstrap settings and seeds, and
  M2's code/environment provenance. Hash the configuration's versioned canonical
  encoding. Store the receipt immutably by digest; editing a local receipt or
  calling `freeze` again cannot rewrite the registered one.
- **Verify before requesting test access.** `micro-train evaluate --final
  --dataset NAME <run>` retrieves the registered receipt and refuses an unfrozen run,
  changed weights or baseline artifacts, or any effective configuration mismatch,
  including CLI overrides and code/environment provenance. Verify and load the
  same immutable artifact snapshots that inference will use; do not hash a path
  and then reopen mutable contents.
  These refusals happen before reservation and before any test bytes are opened.
- **Atomic reservation outside feature branches.** All evaluators use one
  protected, durable authority shared by branches, clones, machines and workers
  on the protected `ledger/test-reads` ref of the configured origin repository.
  Its repository identity is trusted configuration, not a caller-selectable
  remote. Manifest registrations, freeze receipts and evaluation events live
  there, independently of feature-branch manifests and history. An atomic
  insert-if-absent on `test_sha256` records
  `{attempt_id, test_sha256, manifest_sha256, freeze_receipt_sha256, reserved_at,
  status: reserved}` before opening test bytes, parsing rows or running inference.
  A durable successful reservation is required; an existing key, unavailable
  authority or uncertain commit outcome fails closed with no test access. There
  is no branch-local/offline fallback. The store must retain reservations from
  failed runs and abandoned branches; no deletion, lease expiry or reset makes
  a consumed hash available again. State transitions and their append-only audit
  events commit atomically.
- **Authoritative ref updates.** Fetch the authority's current tip, validate the
  requested append against that complete history, and create a commit whose
  parent is exactly that tip. Push without force using that expected old ref
  value; a changed tip rejects the update. Fetch and revalidate after a rejected
  push, refusing if the hash is already reserved. Required server-side checks
  validate append-only history, unique reservations and legal state transitions
  before accepting an update. Protection forbids force-pushes and deletion;
  missing protection or validation fails closed. A non-force push alone does not
  validate event contents. A report and its completion event are committed
  together in one accepted ref update.
- **One start, one pass.** Only the reservation winner may atomically transition
  its attempt from `reserved` to `running`; this transition grants one evaluator
  permission to open the split after an acknowledged durable update. An uncertain
  start outcome permits no test access and no repeat start. A retry of a
  reservation RPC may inspect the same attempt but cannot grant another start.
  First verify the complete raw-byte hash
  against the trusted pin without parsing or inference, using bounded buffers.
  Then stream rows from that same immutable input snapshot, checking overlap
  against the union of the group's frozen train/val row hashes and checking
  labels before scoring each batch. Raw integrity verification is part of this
  recorded attempt; it does not authorize a second parsing/scoring pass. The
  entire frozen group is scored together in that pass, not by separate final
  invocations for each baseline or comparison.
- **Failures consume the reservation.** Malformed JSON, unknown labels, overlap,
  digest mismatch, inference errors and crashes after reservation all leave the
  hash consumed. Append a terminal `failed` event when possible; a crash leaving
  `reserved` or `running` still blocks all retries. Never resume test parsing or
  inference after interruption, even for the same receipt. Inspecting status or
  retrieving an already committed report is allowed without reopening the split.
  A replacement split needs a newly registered, distinct hash.
- **Publish only committed results.** Keep row content, predictions and partial
  scores out of stdout, logs and reports. Only after all input and disjointness
  checks pass does the authority durably commit the complete report and terminal
  `succeeded` event bound to the reservation and freeze receipt; then expose the
  report. Completion is idempotent and never repeats inference. A failed attempt
  publishes failure metadata, not scores.
- **Git ledger is an evidence mirror.** Export authority events, including failed
  and incomplete attempts, to `results/micro/<dataset>/test-ledger.jsonl`. CI
  retains the base-branch append-only check and verifies exported receipt/event
  identities against the authority; missing history or a forged success cannot
  make a score reportable. Git history cannot grant or release reservations.
- **Stated limit.** This enforces one recorded attempt through the authorized
  evaluator, not detection of reads outside it. The report names that limit and
  the near-duplicate limit. M3 is not complete without the shared authority and
  the acceptance tests below; the repository layout audit alone does not certify
  this protocol.
- Metrics: accuracy, macro-F1, per-class precision/recall, confusion matrix,
  expected calibration error; 95% bootstrap intervals on accuracy and macro-F1.
- Baselines on the same splits: majority class, and byte n-gram logistic
  regression, selected on train/val and frozen into the same group before the
  final attempt.

**Exit criteria.** Use synthetic test fixtures, an instrumented test opener,
parser and inference call counters, and two clones of a bare origin with the
authority ref's production validation and protection rules enabled. Verify those
rules on the deployed authority as well. The following acceptance tests must
pass; assertions cover ordering and durable state, not just exit codes.

| Attempt / fault injection | Required observation |
| --- | --- |
| Same file for val/test, identical bytes at two paths, or a shared normalised row in train/test or val/test | Refused; any test-byte read is preceded by reservation; failures discovered after reservation keep the hash consumed and publish no score. |
| Unfrozen run; mutated `model.safetensors`, baseline artifact, config, label order or CLI selection override; edited local freeze receipt | Refused before reservation; zero test opens, parses and inference calls. |
| Swap a model/config path between verification and load | Either rejected before test access or the verified immutable snapshot alone is evaluated; substituted artifacts never reach inference. |
| Missing/untrusted manifest, caller-chosen test hash, or a changed test object at the pinned path | No caller pin is accepted; byte mismatch is detected after reservation but before parsing/inference; the hash remains consumed and no score appears. |
| Repeat `--final` after success or failure, using another path, dataset alias, run, model, branch or clone | Reservation refused before the test opener, parser or inference is called. |
| Malformed JSON or an unseen label after valid earlier batches; overlap found late; inference exception | Reservation precedes the first open/parse/inference; terminal failure stays recorded; no partial predictions or scores escape; another invocation cannot reread. |
| Kill the process just after reservation, during evaluation, or before report commit | Durable reservation remains visible from another process; no expiry or retry permits another start or test read; no success report is published. |
| Authority unavailable, or connection lost around reservation/start commit | No test access without acknowledged durable reservation and start; if committed, the key remains consumed; status/RPC retries cannot create a second start. |
| Two synchronized processes on distinct feature branches/clones reserve the same test hash against the shared authority | Exactly one reservation and at most one start succeeds; the loser has zero test opens/parses/inference; abandoning the winner's branch does not release the key. |
| Two workers try to start the same attempt, including a repeated successful reservation RPC | Exactly one `reserved` to `running` transition; at most one parsing/scoring pass. |
| Lose the reply after a successful report commit, then retry completion or retrieve results | The same stored report and terminal event are returned; zero additional test opens or inference calls. |
| Edit/delete a mirrored ledger line, omit an authority failure, forge a receipt/success event, or use a local fallback store | CI/report publication refuses the evidence; the authority's consumed key is unaffected. |
| Push an edited/deleted authority event, a duplicate reservation or an illegal transition; force-push/delete the authority ref; disable its required validation or use another origin | The protected authority rejects invalid updates; final evaluation refuses absent protections or an untrusted authority; zero test access. |
| Successful frozen model-plus-baselines group | One reservation and one scoring pass produce a complete report bound to unchanged artifact/config digests and the authority's success event. |

On a fixed dataset the report shows the model and both baselines with intervals,
the authoritative reservation, freeze receipt and completion event it came from,
and plainly whether the model's interval clears the stronger baseline's.

**Evidence.** Acceptance-test results and replayable authority event receipts;
`results/micro/<dataset>/report.json`, the mirrored ledger and a rendered table.

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

- **Streaming input.** After M3's reservation, raw integrity verification uses
  bounded buffers; the subsequent JSONL pass reads line by line into fixed-size
  batches from the same immutable snapshot. Neither pass holds the full split.
  The unseen-label check runs on the fly against the training label set (labels,
  not rows).
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
  each. Freeze the complete comparison group before its single M3 final attempt;
  do not spend the same test hash separately for each size, variant or seed.

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
number; its authoritative reservation and committed report make that number
reportable. M7 needs only M2 and can run whenever a GPU is available.
