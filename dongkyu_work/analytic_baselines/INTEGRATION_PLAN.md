# 통합 설계와 완료 기준

이 파일의 경로·옵션은 **구현 예정 사항**이다. 사전 준비 브랜치에서 해당 실행 기능이 제공되는 것은 아니다.

## 파일 구성

| 위치 | 예정 변경 |
| --- | --- |
| `dongkyu_work/analytic_baselines/implementation/heads.py` | AIR, GACL, SLDA의 독립 통계 분류기. 공식 식·dtype·초기화 출처를 함수별 기록 |
| `dongkyu_work/analytic_baselines/implementation/model.py` | FlyGCL 공통 pretrained 백본 + 해당 분류기. 백본 고정 및 eval mode 유지 |
| `dongkyu_work/analytic_baselines/implementation/trainer.py` | 공통 데이터 루프와 통계 갱신의 어댑터. optimizer 없는 setup/report 처리 |
| `methods/{air,gacl,slda}.py`, `models/{air,gacl,slda}.py` | 기존 작업 방식과 같은 얇은 import 연결 |
| `methods/__init__.py`, `models/__init__.py` | `air`, `gacl`, `slda` 등록 |
| `configuration/config.py` | 확장 차원, ridge gamma, SLDA shrinkage/covariance 설정, 통계 저장 device 옵션 |
| 전용 실행 스크립트 | 공통 Si-Blurry 인자 전달, 단일 GPU 명시, `"$@"`로 사용자 옵션 전달 |

기존 `methods/_trainer.py`의 데이터 스트림·평가 스케줄·결과 집계는 재사용한다. 통계형 모델의 setup와 report는 전용 subclass에서 처리한다. 기존 MISA/SinglePrompt 동작은 수정 대상이 아니다.

## 공통 비교 조건

- `utils/train_utils.select_model()` → 새 MODELS 항목으로 연결한다. 백본은 `models.vit`의 기존 로더를 우선하고 지원되지 않는 이름은 기존 timm 경로를 따른다. 별도 DeiT/ResNet checkpoint를 몰래 사용하지 않는다.
- 특징은 분류 head 이전의 벡터를 사용한다. 실제 `models.vit`의 반환 형태를 확인하여 CLS/평균 pooling을 명시적으로 선택하고, 모든 세 방법이 같은 특징을 받게 한다. 특징 차원을 읽어서 분류기를 생성한다.
- 기존 `OnlineSampler`, `OnlineTestSampler`, `n/m/rnd_NM`, seed, 데이터 전처리, 평가 간격과 결과 파일 구조를 유지한다.
- `exposed_classes` 순서로 외부 label을 내부 index로 바꾼다. AIR/GACL의 증가형 출력도 전체 클래스 형식에 맞춰 연결한다. 최종 평가 후보는 seen-class mask로 제한한다.
- `online_step(images, labels, idx)`는 `(loss, accuracy)`를 반환한다. 평가는 `avg_loss`, `avg_acc`, `cls_acc`를 반환한다. CE는 공통 기록용 평가 loss이며 analytic 학습 목적함수를 CE로 바꾸지 않는다.
- task 시작/종료에서 기존 통계나 랜덤 확장 행렬을 재초기화하지 않는다. 데이터 label의 재등장은 동일 클래스로 누적한다.
- 최초 구현은 단일 GPU에서 검증한다. 통계형 모델의 multi-GPU는 DDP gradient 동기화만으로 구현되지 않는다. 지원 전에는 명확히 거부하고, 추후 global batch/statistics 집계와 중복 누적 방지를 검증해야 한다.

## 방법별 보존 항목

### GACL

- 공식 `methods/GACL.py::fit()`의 R/W 재귀 갱신과 새 출력 열의 zero padding을 보존한다.
- 공식 실행 스크립트 기준 확장 차원 5,000, gamma 100, online_iter 3은 출처 있는 설정이다. 설정 파일에 노출하고 사용자 변경을 허용한다.
- 랜덤 linear expansion + ReLU는 학습하지 않고 seed로 고정한다. 공식 GACL은 float32 확장 후 float64 통계를 계산하므로, AIR의 dtype 선택과 무조건 합치지 않는다.
- 공식 `online_evaluate`에는 FlyGCL이 요구하는 `avg_loss`가 없고 task_id/end 인자도 없으므로 연결을 보완한다.

### AIR

- 원본 `GeneralizedARM`의 클래스별 A, C, cnt와 가중치 `1/cnt`의 정규화를 보존한다. 단순히 현재 배치만 reweight하면 과거 클래스 가중치 변화가 반영되지 않아 다른 알고리즘이 된다.
- 공식 코드의 `fit()`는 통계만 갱신한다. 새 통계가 생기면 classifier를 dirty로 표시하고 예측 전에 `update()`를 수행한다. 검증 전체를 대상으로 통계를 갱신하면 안 된다.
- float64 통계를 유지하고 mixed precision 대상에서 제외한다. 메모리 절약을 이유로 임의의 대각 근사나 float32 전환을 기본값으로 넣지 않는다.
- 클래스별 합을 순회하여 임시 `[C,D,D]` 텐서를 줄일 수 있지만, 수학적 동등성 테스트가 필요하다. 이 방법도 영구 저장 A의 `C*D*D*8` 바이트 자체를 없애지는 않는다.
- 원본의 cnt는 등록 buffer가 아니므로 일반 `state_dict()`에 없다. 이식 시 cnt까지 저장·복원하고 결과가 이어지는지 확인한다.
- bias=False가 공식 기본값이다. bias 지원을 확장할 경우 원본의 bias 분기 차원/device를 별도로 점검한다.

### SLDA

- 원저자의 `StreamingLDA.fit()`와 `predict()`가 기준이다. 클래스별 mean/count, 공유 Sigma, shrinkage, Lambda 캐시를 유지한다.
- 배치 특징을 추출한 후 샘플별 순서대로 갱신한다. 동일 label이 여러 번 들어 있는 배치를 단일 `fit()` 호출에 넘기지 않는다.
- GACL의 첫-task 반복 `fit_base()` 경로는 채택하지 않는다. 전체 stream을 먼저 읽어 초기화하는 것도 금지한다. 기본 후보는 원저자 생성 상태에서 sample-wise streaming이며, base 초기화 모드를 제공한다면 이미 관측한 prefix만 사용하는 정책을 명시한다.
- 원저자 코드의 `self.device='cuda'`와 CPU score 반환을 프레임워크 device 정책에 맞춘다. 통계·카운터를 저장하며 복원 후 precision 캐시를 무효화한다.

## online_iter와 데이터 사용량

`online_iter`는 gradient 모델에서 batch를 여러 번 학습하는 용도로 쓰이지만, 통계 모델에서 반복하면 같은 관측을 재누적한다. ridge 정규화 대비 데이터 가중치와 SLDA 통계가 달라진다. 조용히 무시하거나 모든 방법에서 임의로 동일 값으로 바꾸지 않는다. 단일-pass 예시는 1을 명시하고, 기존 GACL의 3회 모드를 지원한다면 반복 누적이라는 의미를 로그와 문서에 남긴다. fractional online_iter는 별도 의미를 설계하지 않은 상태에서 허용하지 않는다.

## 통합 완료 판단에 필요한 검증

1. 현재 준비한 reference-core 검증에 새 adapter를 추가하여 같은 특징·label·초기 상태의 출력/통계를 대조한다.
2. 비연속 외부 label, 재등장 클래스, 한 클래스뿐인 첫 batch, task 경계, unseen logit가 최대인 입력을 확인한다.
3. 실제 새 model/trainer registry로 공통 `main.py` 흐름을 작은 합성 입력으로 실행하여 학습 → 중간 평가 → task 종료 → 결과 저장을 확인한다. mock 사용 위치를 기록한다.
4. `state_dict` 복원 이후에도 후속 누적과 평가 결과가 동일한지 확인한다.
5. 실제 공통 ViT의 특징 shape, checkpoint 로드, 백본 동결을 확인한다. CPU 특징-only 검증으로 이를 대체했다고 쓰지 않는다.
6. GPU 실행 확인 여부와 성능 실험 여부는 별도로 기록한다. 현재 요청의 사전 준비에는 논문 성능 재현을 포함하지 않는다.

## 출처와 라이선스

`SOURCE_MANIFEST.json`에 모든 선택 파일의 원본 commit/blob을 고정했다. AIR 저자 통합 코드와 Deep_SLDA에는 MIT LICENSE가, GACL에는 GPL-3.0 LICENSE가 동봉되어 있다. 원본 경로별 라이선스를 유지하고 이식한 코드에는 출처를 표시한다.
