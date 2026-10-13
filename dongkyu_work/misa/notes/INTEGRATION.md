# MISA downstream Model + Trainer integration (Phase 2a)

Implementation based on Kang et al., **MISA: Mask and Initial-Session Adaptation** (ICLR 2025).
Source snapshot: `kangzhiq/MISA@72e7c960c73eceb1e3f63950d18ad6baff52b560`
([vendored upstream](../original/MISA/), [differences](./MISA_vs_FlyGCL.md)).

## Actual edits

| File | Purpose |
| --- | --- |
| `dongkyu_work/misa/implementation/model.py` | Dedicated MISA model adapter: official downstream hyperparameters (g prompt 10, e prompt 20, e pool 10), required ISA prompt loading, checkpoint type/shape checks |
| `dongkyu_work/misa/implementation/trainer.py` | Dedicated MISA trainer: reuse FlyGCL's Si-Blurry loop and batch-wise class mask during training, but **do not add seen-class mask at evaluation** (upstream behavior) |
| `models/__init__.py` | Register `MODELS["misa"]` |
| `methods/__init__.py` | Register `METHODS["misa"]` |
| `configuration/config.py` | New `--misa_prompt_dir` option; `--load_pt` remains for legacy DualPrompt |
| `dongkyu_work/misa/scripts/run_misa.sh` | Example execution config (not executed) |

| `models/misa.py`, `methods/misa.py` | Thin import-only compatibility shims; required for the root FlyGCL registries |
| `dongkyu_work/misa/checkpoints/` | Official g/e prompt binaries committed unchanged by upstream blob SHA; CPU shape verifier available |

## Reused vs modified

- Official `scripts/misa.sh` uses `--mode DualPrompt --load_pt`: MISA is a two-stage approach, but its **downstream model is DualPrompt with ISA-pretrained prompts**.
- Reuse: FlyGCL `models/dualprompt.py` for prompt insertion, cosine-similarity selection, CLS representation, and loss `CE + similarity`; FlyGCL `methods/dualprompt.py` for online training and batch logit mask; FlyGCL `_Trainer` for the sampler, augmentation, logs, metrics, task boundaries.
- Override: fixed official prompt layout, tensor-shape-checked pretrained loading, **raw-logit evaluation**.
- Disable optional unrelated RPFC/EMA/MePo features to avoid altering the MISA baseline.
- Prompt index selection implementation originates from FlyGCL's repaired global-slice indexing (vs the original upstream implementation). Therefore not claimed to be byte-for-byte or numerically identical to upstream.

## Official ISA pretrained prompts: uploaded (2026-10-13)

- `dongkyu_work/misa/checkpoints/g_prompt.pt` — official blob SHA `b21ade5ab36b97346a3e7a5b54e9ef6aaaffed53` (62,317 bytes).
- `dongkyu_work/misa/checkpoints/e_prompt.pt` — official blob SHA `3b88950a0cabbdd533357fb84f83445fd1f47463` (1,844,077 bytes).
- Upstream: `kangzhiq/MISA@72e7c960c73eceb1e3f63950d18ad6baff52b560`; binary contents are referenced by **exact original Git object SHA**.
- MISA ViT-B/16 implementation expected shapes: g `(1,20,768)`, e `(10,60,768)`; original g tensor pickle metadata statically confirmed. Actual CPU PyTorch load of both tensors has not been executed here.
- Run `python dongkyu_work/misa/scripts/verify_prompts.py` in an environment with PyTorch installed to validate **the real loaded tensor contents, dtype and shapes**; this script performs no training.
- See [checkpoint verification details](../checkpoints/README.md). `--method misa` loads these weights by default, so `--load_pt` is not needed. Optional override: `--misa_prompt_dir`.

## Example: not run

```bash
python main.py --method misa --dataset cifar100 --n_tasks 5 --n 50 --m 10 \
  --rnd_NM --batchsize 64 --online_iter 3 --num_epochs 1 --lr 0.005 \
  --opt_name adam --sched_name default --eval_period 1000 \
  --transforms autoaug --misa_prompt_dir ./dongkyu_work/misa/checkpoints
```

## Verification scope

- Mock-based, **CPU-only** checks conducted on proposed code: missing/incorrect-shape checkpoint rejection, model constructor fixed prompt layout, raw-logit evaluation without evaluation mask.
- Integration source/registry/CLI paths reviewed statically.
- **Not done:** importing the *full* FlyGCL runtime with actual ViT backbone and official tensors; no dataset run, no GPU training, no numerical equivalence/reproduction claim.
- **Not implemented:** ISA/FAM pretraining integration and generation of new official initial-session prompt checkpoints. Reuse of provided ISA pretrained prompt tensors is the downstream comparison path.

Phase 2a goal: ready-to-register model/trainer, with checkpoint acquisition completed; CPU tensor load and full runtime validation remain outstanding.

## Directory organization (post-reorganization)

All MISA-specific implementations and scripts are in `dongkyu_work/misa/`; only the minimal root-level import shims and registry/CLI glue remain in `models/`, `methods/`, and `configuration/` so that `python main.py --method misa` still resolves. No algorithmic code was changed during the relocation. The `run_misa.sh` entry point is now `bash dongkyu_work/misa/scripts/run_misa.sh` from the repository root.
