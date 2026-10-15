# 공식 MISA vs FlyGCL: 초기화 · Forward · Gradient 직접 검증

**결과: CPU 수치 동등성 검증 PASS** (2026-10-15).  
[GitHub Actions SUCCESS](https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38671110633)  
**브랜치:** `audit/misa-official-parity` (이 검증을 위해 실제 MISA 구현, 공통 모델, Trainer는 수정하지 않음).  
**스크립트:** [`verify_official_init_forward.py`](../scripts/verify_official_init_forward.py)

## 비교 대상

1. 공식 MISA 스냅샷의 **실제 Python 클래스 정의** `original/MISA/models/dualprompt.py::Prompt, DualPrompt` (AST 그대로 로딩, method body 수정 없음).
2. FlyGCL `implementation/model.py::MISA` (현재 사용되는 model implementation).
3. **독립 Backbone 구현:** `original/MISA/models/vit.py::VisionTransformer`을 독립 import하여 FlyGCL `models/vit.py::VisionTransformer`와 비교.
4. 공식 제공 가중치 `checkpoints/g_prompt.pt`, `checkpoints/e_prompt.pt`를 두 모델에 각각 로딩.

## 성공한 항목 (CI 출력)

```text
PASS independently constructed ORIGINAL and FlyGCL ViT classes: 152 tensors exact
PASS architecture/Prompt config (12 ViT blocks, 768 features)
PASS seeded backbone including FC init: 152 tensors exact
PASS independent seeded E-Prompt key and FC initialization; official ISA G/E weights
DIFF official defines scale/translation; adapter omits them
PASS evaluation task 0: raw 100-class logits
PASS training task 0: logits/loss/5 param gradients
PASS evaluation task 1: raw 100-class logits
PASS training task 1: logits/loss/5 param gradients
PASS official MISA vs FlyGCL independent seeded initialization and full ViT parity
LIMIT same offline-random ViT factory; real upstream .npz loading and GPU/accuracy untested
```

### 재현성 조건

- Python 3.10, PyTorch 2.5.1+cpu, timm 0.6.12 (GitHub Actions Linux CPU).
- **독립 ViT 클래스 초기화:** 공식 ViT와 FlyGCL ViT를 각각 같은 seed로 생성해 동일한 152개 `state_dict` 텐서(FC weight/bias 포함)가 **정확히 같음**을 검사.
- **MISA 전체 모델 초기화:** 공식 모델과 통합 모델을 각각 `seed=1993`으로 새로 생성. 두 모델에 동일한 **랜덤 ViT-B/16 생성 함수**를 각각 호출하도록 사전학습 로더를 테스트에서만 대체. 모든 152개 백본/FC `state_dict` 텐서, 분류기 weight/bias, E-Prompt key를 실제 생성 직후 직접 비교 (**모델 간 파라미터 복사 없음**).
- 공식 원본은 `load_pt=True` 흐름을 사용. 고정 상대 경로만 테스트 안에서 보존된 `checkpoints/g_prompt.pt`, `e_prompt.pt`로 변경. 통합본은 생산 코드의 정상 체크포인트 로더 사용.
- **모델 Forward/Gradient:** 실제 12-block 224×224 2개 합성 이미지. 평가 `task_id=0,1`에서 raw 100-class logits, similarity 및 expert counter 일치. 학습 `task_id=0,1`에서 미니배치 mask(-inf), CE+similarity loss 및 `backbone.fc.weight`, `backbone.fc.bias`, `g_prompt.prompts`, `e_prompt.key`, `e_prompt.prompts`의 Gradient 일치.
- Gradient tolerance `rtol=atol=5e-4`, logits/loss tolerance `rtol=atol=2e-5`; 초기값은 `rtol=atol=0`.

## 발견된 한 가지 구조 차이

원본 MISA는 `load_pt=True`일 때 추가 `scale`, `translation` 파라미터를 등록하지만, 현재 FlyGCL 통합본은 해당 파라미터를 만들지 않습니다. 공식 기본 경로에서 둘 다 Forward에 연결되지 않고 이번 backward에서도 `grad is None`임을 확인했습니다. **현재 기본 방식의 출력·학습 Gradient 차이는 확인되지 않았습니다.** 원본의 추가적인 G_dist 등 비기본 모드를 재현한다는 의미는 아닙니다.

## 아직 검증하지 않은 것과 비교 시 주의사항

- **ImageNet-21K 사전학습 ViT `.npz` 실제 로드:** 양쪽에 동일한 랜덤 ViT 생성 함수만 사용하여 비교함. 직접 로드된 가중치 완전 동등성은 별도 GPU/환경 검증 필요.
- **실제 CIFAR-100 GPU 학습 및 최종 정확도:** 수행하지 않음.
- **평가 마스킹:** 공식 MISA는 raw logits 평가. FlyGCL의 통합 MISA는 seen-class mask 평가. 이번 Forward 동등성 검사는 **평가 마스크 적용 전 원시 모델 logits** 기준임. 공정 비교를 위한 공통 평가와 공식 논문의 재현 수치를 혼동하지 않아야 함.
- **ISA Prompt 추가 사전학습 사용:** 공식 알고리즘 구성요소로 유지했으며, 다른 baseline과 사전학습 예산이 동일하다는 뜻이 아님.

## 종합 판단

**공식 MISA 모델의 기본 Forward/학습 수식은 위 테스트 범위에서 FlyGCL 통합과 수치 동등성이 확인되었습니다.** 초기 백본 클래스, E-Prompt Key, classifier 초기값도 동일 seed에서 일치합니다. **실제 사전학습 백본 로더 및 데이터 기반 최종 성능까지 재현되었다고 주장하지 않습니다.** 다른 비교군과의 '공정 비교'는 본 알고리즘 검증 외에도 사전학습 사용·공통 Si-Blurry 스트림·평가 프로토콜·학습량의 보고를 필요로 합니다.
