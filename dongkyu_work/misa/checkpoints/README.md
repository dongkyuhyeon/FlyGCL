# Official MISA initial-session pretrained prompt checkpoints

2026-10-13 기준, 두 파일을 **공식 kangzhiq/MISA의 원본 Git Blob SHA 그대로 Git에 업로드**했습니다.

| 파일 | 크기 (bytes) | 공식 Git blob SHA | PyTorch CPU로 확인한 tensor shape |
| --- | ---: | --- | --- |
| [g_prompt.pt](g_prompt.pt) | 62,317 | `b21ade5ab36b97346a3e7a5b54e9ef6aaaffed53` | `(1, 20, 768)` |
| [e_prompt.pt](e_prompt.pt) | 1,844,077 | `3b88950a0cabbdd533357fb84f83445fd1f47463` | `(10, 60, 768)` |

Upstream: [MISA official pretrained_prompt](https://github.com/kangzhiq/MISA/tree/72e7c960c73eceb1e3f63950d18ad6baff52b560/pretrained_prompt)

## 검증 완료: 실제 PyTorch CPU 로딩 성공

- **증빙:** [GitHub Actions 검증 작업](https://github.com/dongkyuhyeon/FlyGCL/actions/runs/38356276565), [상세 검증 기록](../notes/CPU_VERIFICATION.md)
- 공식 Git SHA 검사, `torch.load(..., map_location="cpu", weights_only=True)`, tensor type, dtype `torch.float32`, 정확한 shape, `torch.isfinite` 모두 **PASS**.
- **실제로 확인한 크기:** `g_prompt.pt=(1,20,768)`, `e_prompt.pt=(10,60,768)`.
- **미검증:** ViT-B/16 전체 모델 생성·Forward, 학습·평가 실험 및 논문 재현.

## 사용 전 CPU 로딩 검사 (데이터 학습/GPU 실험 없음)

Git 저장소 루트에서:

```bash
python dongkyu_work/misa/scripts/verify_prompts.py
```

이 스크립트가 공식 SHA를 다시 계산하고, `torch.load(..., map_location="cpu", weights_only=True)`로 직접 파일을 열어 tensor 여부, dtype, shape를 검사합니다. 성공하면 MISA 모델 `_load_misa_prompts()`에 필요한 **체크포인트 텐서 형식 검증**이 끝납니다. 단, ViT Forward 검증과는 구분해야 합니다.

실행: `python main.py --method misa --misa_prompt_dir ./dongkyu_work/misa/checkpoints ...`

공식 가중치 파일을 임의로 변형하거나 이름만 같은 파일로 대체하지 마십시오.
