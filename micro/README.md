# micro-train

A micro LLM / text-classifier training pipeline, provisioned as a podman
container by the repository's `flake.nix`.

Models are small transformers trained from scratch on UTF-8 bytes (vocabulary
256), so nothing is downloaded at train time and any script or symbol has one
encoding. Two modes:

| Mode | Input | Model | Metric |
| --- | --- | --- | --- |
| `lm` | a text file | causal transformer, next-byte prediction | held-out bits per byte (last 10% of the corpus) |
| `classify` | JSONL `{"text", "label"}`, train and eval files | bidirectional encoder, mean-pooled, linear head | eval accuracy |

Each run writes `model.safetensors`, `config.json` and `metrics.json`; the
same arguments on the same device give the same metrics.

## Run it in podman

```sh
nix run .#run -- lm --data corpus.txt --out runs/lm --steps 2000
nix run .#run -- classify --train train.jsonl --eval eval.jsonl --out runs/clf
nix run .#run -- lm --help
```

`.#run` loads the image into podman if it is not there yet, then runs it with
the current directory mounted at `/work` (`:Z` for SELinux, `--userns=keep-id`
so outputs belong to you). Paths in the arguments are relative to `/work`.
The image tag is the image's content hash, so a run always uses exactly the
image this flake built; `nix run .#load` only loads it.

Model size flags: `--d-model 128 --layers 4 --heads 4 --context 256` (defaults,
about 0.9M parameters); `--steps`, `--batch-size`, `--lr`, `--seed`.

## GPU

`nix run .#run-cuda -- ...` builds the image with `torch-bin` (upstream CUDA
wheels, unfree; nothing CUDA is compiled locally) and passes
`--device nvidia.com/gpu=all`. The host needs the NVIDIA driver and a CDI
spec from the NVIDIA Container Toolkit
(`nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`). The CPU image is
the default and needs nothing from the host but podman.

## Develop

```sh
nix develop          # python + torch + pytest + podman, PYTHONPATH set to micro/
pytest micro/tests   # also run inside every nix build of the package
nix flake check      # builds the package (with its tests) and the image
```
