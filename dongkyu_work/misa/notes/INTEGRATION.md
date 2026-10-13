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
- MISA-only `MISAExpertPrompt` reproduces upstream released **local-index gather** (not FlyGCL's global-index fix). Direct CPU parity with upstream Prompt.forward passed; no real-dataset numerical equivalence claimed.

## Official ISA pretrained prompts: uploaded (2026-10-13)

- `dongkyu_work/misa/checkpoints/g_prompt.pt` — official blob SHA `b21ade5ab36b97346a3e7a5b54e9ef6aaaffed53` (62,317 bytes).
- `dongkyu_work/misa/checkpoints/e_prompt.pt` — official blob SHA `3b88950a0cabbdd533357fb84f83445fd1f47463` (1,844,077 bytes).
- Upstream: `kangzhiq/MISA@72e7c960c73eceb1e3f63950d18ad6baff52b560`; binary contents are referenced by **exact original Git object SHA**.
- **CPU loading verified (2026-10-13):** both tensors successfully loaded via PyTorch 2.5.1+cpu; g `(1,20,768)`, e `(10,60,768)`, both float32 and finite. [Proof and verbatim log](CPU_VERIFICATION.md).
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

Phase 2a goal: registered MISA, official Prompt checkpoint CPU validation passed. Stage 6 full ViT model structure Forward/Backward, Stage 7 shared Trainer and Stage 8 synthetic `main.py` entrypoint checks have since passed. Actual pretrained ViT + real dataset / CUDA GPU run remains unverified.

## Directory organization (post-reorganization)

All MISA-specific implementations and scripts are in `dongkyu_work/misa/`; only the minimal root-level import shims and registry/CLI glue remain in `models/`, `methods/`, and `configuration/` so that `python main.py --method misa` still resolves. No algorithmic code was changed during the relocation. The `run_misa.sh` entry point is now `bash dongkyu_work/misa/scripts/run_misa.sh` from the repository root.

## Algorithm fidelity and shared protocol

- See [PARITY_PROTOCOL.md](PARITY_PROTOCOL.md): official MISA uses local topk indices in Prompt gathering, differing from FlyGCL Prompt's global-slice offset. The MISA-only implementation preserves released upstream behavior.
- Shared sampler/evaluation schedule/metrics are provided by FlyGCL _Trainer; note method-specific inference mask differences.

- CPU parity CI **passed**: [run 38357870025](https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38357870025). [Verified checks](PARITY_VERIFICATION.md): official Prompt selection, training mask/loss, MISA raw-logit evaluation, synthetic Si-Blurry sampler determinism, shared Trainer wiring. Stage 6 full ViT structure forward/backward passed separately, while full pretrained runtime remains untested.

## 6단계: 실제 ViT-B/16 Forward/Backward CPU 검증

- GitHub Actions [성공한 실제 실행 로그](https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38362437114) — Python 3.10, PyTorch 2.5.1 CPU, timm 0.6.12.
- [상세 기록](FORWARD_VERIFICATION.md): 실제 ViT-B/16 12블록 모델 생성, 공식 G/E Prompt 삽입, 합성 데이터 Forward/CE+similarity loss/Backward, Task 1 진입 후 Eval Forward **모두 PASS**.
- **테스트 한계:** 인터넷에서 `.npz`를 다운로드하지 않도록 **백본만 랜덤 초기화**한 CPU 구조 검증. 공식 Prompt는 실물 사용. Full CIFAR-100 Trainer 실행/정량 성능 재현은 수행하지 않음.

## 7단계: 공유 Trainer 온라인 업데이트 연동 CPU 검증

- **SUCCESS:** [GitHub Actions 실행 38364210845](https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38364210845). [검증 보고서](TRAINER_VERIFICATION.md), [검증 스크립트](../scripts/verify_trainer.py).
- 공통 Si-Blurry OnlineSampler/DataLoader를 사용한 100-class 가상 이미지 입력, MISA의 Logit batch mask, 모델 Loss, Backward, Adam.step, 파라미터 변경, OnlineTestSampler/raw-logit 평가, Task 0→1 전환 후 두 번째 업데이트를 모두 통과했습니다.
- GPU/전체 사전학습 ViT 대신 작은 proxy 백본을 사용한 **연결 시험**이므로 실제 `main.py` GPU runner·정량 지표 재현은 별개로 남습니다. 기존 비교군 소스 변경 없음.

## 8/9단계 최종 확인 (2026-10-13)

- [8단계 검증 및 실제 CI 로그](ENTRYPOINT_VERIFICATION.md): `main.py` 실행 경로, 5 Task, 51 updates, 20 evals, shared result `.npy` files; launcher path-independent and missing-Prompt preflight checks.
- [9단계 최종 코드 감사/박사님 보고](FINAL_REPORT.md) 및 [재검증 스크립트](../scripts/verify_final_audit.py): 공식 원본 60개, 기존 비교군 공통 파일 11개 무변경; 원본 parity와 알고리즘별 eval mask 예외 기록.
- GPU 및 실제 `pretrained=True` 백본 다운로드의 완전 실행은 실험 범위 밖이며 완료로 오인하지 않습니다.
