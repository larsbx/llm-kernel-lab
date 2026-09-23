#!/usr/bin/env bash
# Round 1 of expert iteration on one GPU box (e.g. a Runpod pod).
#
#   1. train a LoRA on the verified Lean Workbook proofs (results/workbook-k4)
#   2. serve the bf16 base and base + LoRA from one MAX server on the GPU
#   3. run miniF2F-test (k = 4) on both, same GPU, same precision, same settings
#   4. pack the adapter and both runs into round1-results.tar.gz
#
# The adapter never sees miniF2F-test (train_lora refuses such rows), so the
# base-vs-LoRA difference on it is the round's measured effect. The earlier
# CPU run (104/244) used a 4-bit model and is not the baseline here.
#
# Re-runnable: a finished step is skipped, and each evaluation resumes.
# Needs: >= 24 GB VRAM with bf16, NVIDIA driver >= 580 for MAX, ~40 GB disk.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-runs/gpu-round1}
K=${K:-4}
BASE=models/deepseek-prover-v1.5-rl-bf16
ADAPTER=adapters/round1
export PATH="$HOME/.pixi/bin:$HOME/.elan/bin:$PATH"
mkdir -p "$OUT"
log() { printf '\n== %s  %s\n' "$(date -u +%H:%M:%S)" "$*"; }

log "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
(( mem >= 23000 )) || { echo "FAIL: need >= 24 GB VRAM, have ${mem} MiB"; exit 1; }
(( driver >= 580 )) || echo "WARNING: driver $driver < 580; MAX will refuse the GPU, so only training will run"

log "toolchains"
command -v pixi >/dev/null || curl -fsSL https://pixi.sh/install.sh | PIXI_NO_PATH_UPDATE=1 bash
command -v lake >/dev/null || curl -fsSL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
pixi install --locked
pixi install --locked -e train-cuda

log "model (bf16 weights, fixed tokenizer)"
pixi run python tools/prepare_model.py --weights safetensors

log "Mathlib build cache"
pixi run lean-setup

if [[ -f $ADAPTER/manifest.json ]]; then
  log "LoRA: $ADAPTER exists, skipping training"
else
  log "LoRA training"
  pixi run -e train-cuda train-lora --data results/workbook-k4/verified.jsonl \
    --base "$BASE" --tokenizer "$BASE" --out "$ADAPTER" 2>&1 | tee "$OUT/train.log"
fi

if (( driver < 580 )); then
  log "stopping before evaluation: MAX needs driver >= 580"
else
  log "MAX: serving base + LoRA"
  setsid pixi run max serve --model "$BASE" --served-model-name deepseek-prover-v1.5-rl \
    --devices gpu --quantization-encoding bfloat16 --max-length 2048 --port 8000 \
    --prefer-module-v3 --enable-lora --no-enable-prefix-caching \
    --lora-paths "round1=$ADAPTER" --max-lora-rank 16 > "$OUT/max-serve.log" 2>&1 &
  server=$!
  trap 'kill -- -"$server" 2>/dev/null || true' EXIT
  for _ in $(seq 360); do
    curl -sf localhost:8000/v1/models >/dev/null && break
    kill -0 "$server" 2>/dev/null || { tail -20 "$OUT/max-serve.log"; echo "FAIL: MAX exited"; exit 1; }
    sleep 5
  done
  curl -sf localhost:8000/v1/models

  for model in deepseek-prover-v1.5-rl round1; do
    log "miniF2F-test, k=$K, model=$model"
    pixi run prove --model "$model" --k "$K" --out "$OUT/minif2f-test-$model" --resume
  done

  log "result"
  pixi run python - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
runs = {m: json.loads((out / f"minif2f-test-{m}" / "summary.json").read_text())
        for m in ("deepseek-prover-v1.5-rl", "round1")}
base, tuned = (set(runs[m]["solved"]) for m in runs)
for m, s in runs.items():
    print(f"{m:26s} pass@{s['k']} = {len(s['solved'])}/{s['problems']} ({s['pass_at_k']:.1%})")
print(f"gained {len(tuned - base)}: {sorted(tuned - base)}")
print(f"lost   {len(base - tuned)}: {sorted(base - tuned)}")
EOF
fi

log "packing"
tar czf round1-results.tar.gz --exclude="$ADAPTER/checkpoints" "$ADAPTER" "$OUT"
ls -la round1-results.tar.gz
log "done: download round1-results.tar.gz"
