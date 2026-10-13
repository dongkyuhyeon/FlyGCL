# MISA FlyGCL 통합 최종 검증 및 박사님 보고용 요약 (2026-10-13)

**개발 목표:** 공식 MISA (ICLR 2025)의 downstream Prompt 초기값·핵심 알고리즘을 보존하면서, 개인 FlyGCL fork의 `python main.py --method misa`로 공통 Si-Blurry benchmark / metric writer에 연결. 기존 비교군 코드는 유지하고, 개인 작업은 `dongkyu_work/misa/`에서 관리합니다.

## 최종 검증표

| 단계 | 확인한 사실 | CI 증빙 / 범위 |
| --- | --- | --- |
| 1–3 | 원본 코드 60개 텍스트 파일 보관, 개인 작업 폴더 정리, MISA 모델·Trainer 독립 등록 | GitHub file tree + source inspection |
| 4 | 공식 G/E Prompt 가중치 원본 SHA, 실제 CPU torch.load, tensor 크기/dtype/finite | [PASS](https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38356276565) |
| 5 | 원본 `Prompt.forward` 10조건 parity, Train Mask, CE+Similarity, Raw-logit eval, 동일 Si-Blurry seed | [PASS](https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38357870025) |
| 6 | 실제 ViT-B/16 12-block 모델 생성, 공식 Prompt 삽입, synthetic Forward/Backward | [PASS](https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38362437114); **backbone random init** |
| 7 | 실제 공통 DataLoader/Trainer/Mask/Loss/Adam.step/Task 전환, 두 번 업데이트 | [PASS](https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38364210845); **small proxy backbone** |
| 8 | `main.py` 전체 진입점, 5 Task, 51 online updates, 20 evals, `.npy` 저장, 런처 path/preflight | [PASS](https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38365379698); **CPU synthetic/proxy** |
| 9 | 기존 공통 FlyGCL 11개 코드와 공식 MISA 원본 60개 파일 무변경 Git SHA 대조, 실행 경로·문서·코드 감사 | `verify_final_audit.py` + [Actions workflow](../../../.github/workflows/verify-misa-final.yml) |

## 코드 감사 결론

- **변경 격리:** FlyGCL의 `main.py`, `models/dualprompt.py`, `methods/dualprompt.py`, `methods/flyprompt.py`, `methods/_trainer.py`, `models/l2p.py`, `utils/onlinesampler.py`, `utils/train_utils.py`, `models/vit.py`, `requirements.txt`, `datasets/OnlineIterDataset.py`는 MISA 통합 이전 커밋 `30b102c26e6c129d6f134a53552a409ad552f2ad`과 **바이트/Blob SHA 동일**합니다. 벤치마크의 기존 실행 코드는 변경하지 않았습니다.
- **원본 보존:** `original/MISA/` 공식 텍스트 60개도 해당 커밋 대비 SHA가 모두 동일합니다.
- **실질 구현 위치:** `implementation/model.py`, `implementation/trainer.py`, `scripts/`, `notes/` 및 `checkpoints/`. 루트의 `models/misa.py`, `methods/misa.py`는 import-only 연결부. 별도 공통 등록부는 `models/__init__.py`, `methods/__init__.py`; MISA 옵션은 `configuration/config.py`에 추가했습니다.
- **공식 MISA 공개 코드 충실도:** upstream Prompt key slice `local topk` gather는 FlyGCL 일반 Prompt의 `topk+s`와 다릅니다. 공식 소스 동작을 복원한 **MISA 전용 `MISAExpertPrompt`**를 사용했고 실측 parity가 통과했습니다. 공식 ISA `.pt` 가중치를 사용하지만 ISA/FAM 사전학습을 재실행하지 않았습니다.
- **비교 공정성 주의:** 공통 Si-Blurry 데이터 스트림/평가 지표를 공유하지만 **MISA는 raw logits 평가, FlyPrompt는 observed-class masking**을 사용하므로 예측 후처리까지 완전히 동일하지 않습니다. 공식 구현을 보존하기 위한 차이이며 성능 영향은 측정하지 않았습니다.
- **실제 실행의 한계:** `pretrained=True` ViT-B/16 `.npz` 다운로드 및 CPU/GPU 전체 Trainer 실행, 실제 CIFAR-100/다른 데이터셋, 분산 학습, 논문 최종 수치는 **미검증**입니다. 단계 6–8은 구조/연결 검증입니다. 스크립트가 존재한다는 이유로 논문 성능 재현 또는 GPU 실행 성공이라고 보고하면 안 됩니다.

## 실행 방법과 추가 환경 조건

권장 환경: **Python 3.10**, `timm==0.6.12`, 프로젝트 `requirements.txt`; GPU 및 실제 CIFAR-100 데이터 다운로드/접근 환경이 필요합니다. Python 3.11에서는 기존 timm 0.6.12 dataclass 오류를 CI에서 확인해 Python 3.10으로 검증했습니다.

```bash
git clone https://github.com/dongkyuhyeon/FlyGCL.git
cd FlyGCL
# Python 3.10 환경에 의존성 설치 후:
python dongkyu_work/misa/scripts/verify_prompts.py
bash dongkyu_work/misa/scripts/run_misa.sh
```

마지막 명령은 **실제 학습**을 시작합니다. 아직 실행하지 않았습니다. 실행 중 공통 ViT 백본은 `models/vit.py`를 통해 `ViT-B_16.npz`를 다운로드/캐시할 수 있으므로, 네트워크 및 저장공간 확보가 필요합니다. `DATA_DIR`, `MISA_PROMPT_DIR`은 환경변수로 바꿀 수 있습니다.

## 박사님 보고용 문안

> MISA 공식 공개 코드를 기존 FlyGCL의 독립 비교군(`--method misa`)으로 통합하고, 공식 ISA Prompt 가중치 및 핵심 Prompt 선택·Logit Mask·Loss·평가 로직을 검증했습니다. 동일 Si-Blurry 프로토콜에서 실행되도록 공유 Trainer에 연결했으며, CPU 합성 데이터로 실제 `main.py`의 5개 Task, 51회 온라인 업데이트, 20회 평가 및 결과 저장까지 확인했습니다. 기존 비교군 코드와 공식 원본은 변경하지 않았습니다. 단, 본 확인은 실행 연결 검증으로, 실제 사전학습 ViT 및 CIFAR-100 GPU 실험과 논문 성능 재현은 수행하지 않았습니다.
