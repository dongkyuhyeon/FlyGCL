# SinglePrompt on FlyGCL

**상태: 1단계 완료 — 공식 소스 분석 및 고정 커밋 원본 보존.**  
현재 `--method singleprompt`는 아직 등록되지 않았으며, CPU/GPU 실행 및 성능 검증도 하지 않았습니다.

- 논문: *Is Prompt Selection Necessary for Task-Free Online Continual Learning?* (CVPR Findings 2026)
- [공식 GitHub](https://github.com/efficient-learning-lab/SinglePrompt)
- [원본 스냅샷](original/README.md) · [SHA 목록](original/SOURCE_MANIFEST.md)
- [핵심 모델/학습/평가 분석](notes/SOURCE_AUDIT.md)

## 핵심 구현 제약

1. 라우팅이 없는 고정 K/V Prefix Prompt + cosine classifier를 유지
2. 공식 학습의 minibatch masking과 FlyGCL 공통 평가용 seen-class masking을 혼동하지 않음
3. 다른 FlyGCL 비교군 및 공유 ViT 소스는 수정하지 않음
4. 2단계부터 `implementation/`에 실제 코드를 추가하고 `--method singleprompt`로 등록

원본 MIT 라이선스를 포함하고 있으며, 원본 파일은 2단계 이후에도 변경하지 않습니다.
