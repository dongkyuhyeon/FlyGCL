# 사전 준비 검증 기록

실행일: 2026-10-23. Python 3.12, PyTorch `2.5.1+cpu`, CPU, 1 thread, seed 23.

## 실제 실행한 검사

| 검사 | 결과 | 범위 |
| --- | --- | --- |
| 원본 파일 무결성 | PASS | 선택한 공식 파일 17개의 Git blob SHA가 manifest와 일치 |
| GACL | PASS | 실제 공식 `fit()`를 AST에서 불러와 실행. 19개 7차원 특징, 3개 클래스의 재등장/새 등장. batch 분할 19 / 1씩 / 3·6·2·8 각각에서 매 prefix의 W와 joint ridge 해 비교 |
| AIR | PASS | 공식 `GeneralizedARM` 직접 import. 같은 크기의 불균형 label stream, 세 batch 분할. 매 prefix의 weight와 클래스별 역빈도 가중 ridge 해 비교 |
| SLDA | PASS | 원저자 `fit`/`predict` 실행. 19개 7차원 특징의 샘플별 평균·카운트 확인, GACL에 복사된 `fit`과 통계 일치, 예측식·캐시 확인 |
| GACL 안의 SLDA 초기화 제어 흐름 | 확인 | 실제 `online_step`에 fit_base 기록 함수를 연결. task 종료 이전 두 batch 모두 fit_base를 호출함. OAS 계산은 이 검사에서 실행하지 않음 |
| AIR 저장 상태 | 확인 | 원본 cnt는 `state_dict()`에 포함되지 않음. 이식 시 별도 등록/저장 필요 |
| 메모리 산식 | 실행 | 클래스별 A 또는 R/Sigma 등 명시한 행렬의 크기를 산술 계산. 실제 peak GPU 사용량 측정 아님 |

GACL/AIR 비교는 float64, rtol=1e-9, atol=1e-10이다. 검증 기준식의 identity도 float64로 구성했다. SLDA의 `pinverse`와 독립 `solve` 비교는 float32 수치 오차를 고려해 rtol/atol=2e-4이다.

## 테스트를 위해 분리한 부분

- GACL은 원래 Trainer·데이터셋 import를 피하려고 class AST를 추출했다. `fit` 수식은 수정하지 않았고 feature expansion은 identity로 연결했다. 랜덤 projection 및 이미지 backbone 검증이 아니다.
- SLDA는 원본 생성자의 hard-coded `self.device='cuda'` 한 곳을 메모리 안의 AST에서 `'cpu'`로 바꿨다. 저장된 원본 파일은 수정하지 않았다.
- SLDA의 원 논문 base-training/OAS 초기화 재현은 검사하지 않았다. 생성 상태에서 sample-wise update를 검사했다.
- AIR의 작은 7차원 입력에서만 수치 검사를 했다. 8,192차원에서 메모리 할당하거나 역행렬을 계산한 것은 아니다.
- 실행 환경에는 NumPy가 없어 PyTorch의 NumPy 초기화 경고가 있었으나 이 검사 경로는 NumPy를 사용하지 않았고 최종 exit code는 0이었다.

## 최종 출력

```text
PASS exact upstream blob hashes: 17 files
PASS GACL original fit: three batch partitions, repeated/new classes, every prefix vs joint ridge
PASS AIR original ARM: three batch partitions, every prefix vs class-weighted ridge
AUDIT AIR count tensor is absent from state_dict; adapter checkpoint handling required

First predict since model update...computing Lambda matrix...
PASS SLDA original per-sample updates: means/counts, GACL-copy parity, score formula/cache
AUDIT GACL SLDA calls fit_base on each batch until task-end; use original author's streaming core
PASS reference-core preparation only; torch=2.5.1+cpu; CPU; no FlyGCL/GPU/accuracy experiment
```

## 아직 확인하지 않은 것

FlyGCL에 새 방법을 등록하는 코드, 공통 Trainer 실행, 실제 이미지/ViT 특징 추출, checkpoint를 이용한 backbone 확인, 학습 결과 파일 생성, GPU 실행 및 논문 정확도 재현은 이번 사전 준비 검증에 포함되지 않는다. 따라서 이 기록으로 **세 방법의 FlyGCL 통합 완료** 또는 **실제 비교 실험 실행 가능 상태**를 주장하면 안 된다.
