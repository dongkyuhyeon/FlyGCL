> **현재 상태:** MISA 독립 모델·Trainer가 구현되어 `--method misa`로 등록되어 있습니다. 실제 구현은 [../implementation/](../implementation/)에서 관리하며 실행 예시는 [../scripts/run_misa.sh](../scripts/run_misa.sh)에 있습니다. 공식 Prompt 가중치 2개는 확보하여 업로드했습니다. 두 파일의 실제 CPU torch.load/SHA/shape/dtype 검증은 GitHub Actions에서 통과했습니다. 전체 ViT 실행은 미검증입니다. [검증 증빙](CPU_VERIFICATION.md). 아래 기록은 통합 **전** 분석을 포함합니다.

# MISA 통합 작업 기록

MISA (ICLR 2025) 공식 코드와 FlyGCL 사이의 차이를 분석한 문서를 보관합니다.

## 1단계 (완료)
- [**MISA_vs_FlyGCL.md**](./MISA_vs_FlyGCL.md): 공식 코드와 현재 FlyGCL의 구조·학습·Prompt·Mask·FAM 차이 분석
- [**공식 코드 보관함**](../original/): upstream 커밋을 고정한 원본 텍스트 소스 및 복사 manifest

## 2단계 (당시 계획, 이후 등록 작업 진행)
- MISA 독립 `--method misa` 등록과 공통 Si-Blurry Trainer 통합
- Pretrained Prompt 가중치 로딩 인터페이스 보완
- 공통 평가 설정 확인

## [과거, 수정 전] FlyGCL에서 --method misa로 실행할 수 없었던 이유
- FlyGCL구조 안에 method/__init__.py에서 실행 가능한 모델을 등록함
- 그리고 configuration/config.py에서는 등록된 모델만 선택할 수 있도록 제한
- 따라서 --method misa를 입력하면 허용되지 않은 모델이라는 인자 오류가 발생

## [과거, 수정 전] FlyGCL에 MISA 관련 코드는 왜 있었는가?
- MISA 관련 기능에 DualPrompt , MISA 사전학습 Prompt 로딩(--load pt) , FAM Optimizer 이 졵재함
- 하지만 독립적인 misa 모델 등록은 없고 MSIA 전체 알고리즘을 통합한게 아니다
- 즉, MISA 관련 기능 일부는 구현되어 있지만, MISA를 독립적인 비교군으로 실행하는 기능은 없음
  
