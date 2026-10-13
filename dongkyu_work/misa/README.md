# MISA 작업 공간 (dongkyu_work/misa)

MISA (ICLR 2025) 비교군을 FlyGCL의 Si-Blurry 공통 프로토콜에 연결하기 위한 **개인 작업 전용 폴더**입니다.

## 폴더 안내

| 경로 | 용도 |
| --- | --- |
| [original/MISA/](original/MISA/) | 수정 없이 보관한 공식 MISA 텍스트 코드 스냅샷 (60개 파일) |
| [original/SOURCE_MANIFEST.md](original/SOURCE_MANIFEST.md) | 공식 Git SHA·출처·미포함 바이너리 기록 |
| [implementation/model.py](implementation/model.py) | FlyGCL에서 쓰는 MISA 모델의 실제 구현 |
| [implementation/trainer.py](implementation/trainer.py) | FlyGCL에서 쓰는 MISA Trainer의 실제 구현 |
| [scripts/run_misa.sh](scripts/run_misa.sh) | MISA 실행 설정 예시 |
| [scripts/verify_prompts.py](scripts/verify_prompts.py) | 공식 SHA 및 tensor dtype/shape CPU 로딩 검증 스크립트 |
| [checkpoints/](checkpoints/) | 공식 ISA Prompt 가중치 2개가 Git에 저장된 폴더 및 검증 안내 |
| [notes/MISA_vs_FlyGCL.md](notes/MISA_vs_FlyGCL.md) | 공식 MISA와 FlyGCL의 코드 차이 |
| [notes/INTEGRATION.md](notes/INTEGRATION.md) | 통합 상세 설명·범위·검증 한계 |

## 루트에 남겨야 하는 최소 연결부

- `models/misa.py`, `methods/misa.py`: 개인 폴더 구현을 import하는 **얇은 연결 파일**만 있습니다.
- `models/__init__.py`, `methods/__init__.py`: 기존 FlyGCL 실행 모델 선택 등록부 (`misa`).
- `configuration/config.py`: `--misa_prompt_dir` 옵션.
- `README.md`: 전체 프로젝트 안내 링크.

완전히 모두 개인 폴더에만 두면 FlyGCL의 기존 `--method misa` 등록 및 실행 경로가 끊어질 수 있어, 필수 연결부만 루트에 유지합니다.

## 실행 예시 (현재 실제 학습은 수행하지 않음)

저장소 루트에서:

```bash
bash dongkyu_work/misa/scripts/run_misa.sh
```

실행 전 공식 가중치 `g_prompt.pt`, `e_prompt.pt`가 `dongkyu_work/misa/checkpoints/`에 필요합니다. 현재 **공식 바이너리 두 개를 원본 Git SHA 그대로 확보·업로드했습니다.** 다만 PyTorch 실제 로딩/모델 생성/전체 실행과 논문 성능 재현은 아직 검증하지 않았습니다.

공식 MISA 원본 코드는 [kangzhiq/MISA](https://github.com/kangzhiq/MISA)이고, **원본 코드는 수정하지 않았습니다**.
