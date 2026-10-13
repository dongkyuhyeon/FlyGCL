# MISA 통합 작업 기록

MISA (ICLR 2025) 공식 코드와 FlyGCL 사이의 차이를 분석한 문서를 보관합니다.

## 1단계 (완료)
- [**MISA_vs_FlyGCL.md**](./MISA_vs_FlyGCL.md): 공식 코드와 현재 FlyGCL의 구조·학습·Prompt·Mask·FAM 차이 분석
- [**공식 코드 보관함**](../original/): upstream 커밋을 고정한 원본 텍스트 소스 및 복사 manifest

## 2단계 (아직 수행하지 않음)
- MISA 독립 `--method misa` 등록과 공통 Si-Blurry Trainer 통합
- Pretrained Prompt 가중치 로딩 인터페이스 보완
- 공통 평가 설정 확인

## 현재 FlyGCL에 있는 misa를 바로 python main.py --method misa ...로 바로 실행할수 없는 이유
- FlyGCL구조 안에 method/__init__.py에서 실행 가능한 모델을 등록함
- 그리고 configuration/config.py에서는 등록된 모델만 선택할 수 있도록 제한
- 따라서 --method misa를 입력하면 허용되지 않은 모델이라는 인자 오류가 발생

## FlyGCL에 MISA 관련 코드는 왜 있음?
- MISA 관련 기능에 DualPrompt , MISA 사전학습 Prompt 로딩(--load pt) , FAM Optimizer 이 졵재함
- 하지만 독립적인 misa 모델 등록은 없고 MSIA 전체 알고리즘을 통합한게 아니다
- 즉, MISA 관련 기능 일부는 구현되어 있지만, MISA를 독립적인 비교군으로 실행하는 기능은 없음
  
