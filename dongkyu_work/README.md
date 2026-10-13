# Dongkyu's FlyGCL workspace

이 폴더는 원본 FlyGCL 코드를 보존하면서 개인 연구 작업을 정리하기 위한 공간입니다.

## 작업 목적

- MISA (ICLR 2025) 공식 저장소와 FlyGCL 기존 구현 차이 분석
- MISA를 FlyGCL의 Si-Blurry 공통 데이터 스트림 및 평가 프로토콜에 맞게 통합 준비
- 공식코드 기반으로 FlyGCL코드안에 통합해서 같은 프로토콜에서 실행 가능하게 준비 
## 원칙

- 원본 FlyGCL 구조(`methods/`, `models/`, `configuration/`, `scripts/`)는 필요한 경우에만 수정합니다.
- 실제 비교군 등록에 필요한 실행 코드는 해당 기존 디렉터리에 연결합니다.
- 현재 단계에서는 GPU 성능 실험을 수행하지 않습니다.

**MISA 공식 코드:** https://github.com/kangzhiq/MISA

**FlyGCL 원본:** https://github.com/AnAppleCore/FlyGCL

