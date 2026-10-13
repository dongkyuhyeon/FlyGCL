# MISA 통합 작업 기록

MISA (ICLR 2025) 공식 코드와 FlyGCL 사이의 차이를 분석한 문서를 보관합니다.

## 1단계 (완료)
- [**MISA_vs_FlyGCL.md**](./MISA_vs_FlyGCL.md): 공식 코드와 현재 FlyGCL의 구조·학습·Prompt·Mask·FAM 차이 분석
- [**공식 코드 보관함**](../original/): upstream 커밋을 고정한 원본 텍스트 소스 및 복사 manifest

## 2단계 (아직 수행하지 않음)
- MISA 독립 `--method misa` 등록과 공통 Si-Blurry Trainer 통합
- Pretrained Prompt 가중치 로딩 인터페이스 보완
- 공통 평가 설정 확인

이번 단계에서는 실제 훈련·정확도 실험을 수행하지 않습니다.
