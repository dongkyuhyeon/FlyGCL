# SinglePrompt on FlyGCL

**상태: 2단계 모델 구현 및 CPU 공식 동작 대조 완료.**  
**미구현:** FlyGCL `--method singleprompt` 등록과 Trainer/Si-Blurry 데이터 스트림 연결 (3단계 예정). GPU/CIFAR-100 정확도 실험도 아직 수행하지 않았습니다.

- 논문: *Is Prompt Selection Necessary for Task-Free Online Continual Learning?* (CVPR Findings 2026)
- [공식 GitHub](https://github.com/efficient-learning-lab/SinglePrompt)
- [공식 원본 스냅샷](original/README.md) · [SHA 파일 목록](original/SOURCE_MANIFEST.md)
- [1단계 알고리즘 분석](notes/SOURCE_AUDIT.md)
- [2단계 CPU 검증 결과 및 한계](notes/STAGE2_VERIFICATION.md)

## 단계 2에서 추가한 파일

| 경로 | 내용 |
|---|---|
| [implementation/model.py](implementation/model.py) | 기존 FlyGCL ViT를 동결한 상태에서 공식 K/V Prefix와 cosine classifier를 구현한 **독립 모델** |
| [scripts/verify_model.py](scripts/verify_model.py) | 공식 코드와 Prefix Attention·전체 ViT 출력·Prompt/FC 기울기 비교 |
| [GitHub Actions](../../.github/workflows/verify-singleprompt-stage2.yml) | Python 3.10, torch 2.5.1+cpu, timm 0.6.12 환경에서 자동 검증 |

기존 `models/vit.py`·`methods/_trainer.py`·다른 비교군은 수정하지 않았습니다. 공식 원본 코드는 `original/SinglePrompt/`에 그대로 보존했습니다.

## CPU 모델 테스트 (저장소 루트)

FlyGCL의 필수 패키지 설치 후:

```bash
PYTHONPATH=. python -m dongkyu_work.singleprompt.scripts.verify_model
```

테스트는 **사전학습 백본 다운로드 없이**, 랜덤으로 초기화한 동일한 전체 ViT-B/16의 파라미터를 복사해 비교합니다. 공식 사전학습 체크포인트 로딩, 실제 데이터·GPU 성능은 검증하지 않습니다.

## 다음 3단계

공통 FlyGCL `_Trainer`·Si-Blurry sampler와 연결하고, `--method singleprompt`를 등록합니다. 공식 학습 batch mask 및 공식 raw-logit 평가와 FlyGCL seen-class 평가를 명시적으로 구분합니다.
