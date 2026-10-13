# Official MISA initial-session pretrained prompt checkpoints

2026-10-13 기준, 두 파일을 **공식 kangzhiq/MISA의 원본 Git Blob SHA 그대로 Git에 업로드**했습니다.

| 파일 | 크기 (bytes) | 공식 Git blob SHA | ViT-B/16 구현의 예상 tensor shape |
| --- | ---: | --- | --- |
| [g_prompt.pt](g_prompt.pt) | 62,317 | `b21ade5ab36b97346a3e7a5b54e9ef6aaaffed53` | `(1, 20, 768)` |
| [e_prompt.pt](e_prompt.pt) | 1,844,077 | `3b88950a0cabbdd533357fb84f83445fd1f47463` | `(10, 60, 768)` |

Upstream: [MISA official pretrained_prompt](https://github.com/kangzhiq/MISA/tree/72e7c960c73eceb1e3f63950d18ad6baff52b560/pretrained_prompt)

## 현재 검증 수준

- **원본 무결성:** Git에 등록한 Blob SHA를 원본 SHA와 직접 대조해 동일함을 확인했습니다. 이는 전체 바이너리의 바이트 동등성을 보장합니다.
- **형태에 대한 정적 근거:** `g_prompt.pt`는 GitHub에서 base64로 조회하여 PyTorch ZIP 내부 `data.pkl` 메타데이터가 `(1,20,768)` FloatStorage를 나타내는 것을 확인했습니다. `e_prompt.pt`는 공식 크기 및 코드 설정에 근거한 **예상** 형태로 기록했으며, tensor를 실제로 열어 shape를 확인한 것은 아닙니다.
- **실제 CPU torch.load 및 모델 Forward:** 현재 도구 환경에서 공식 대형 바이너리를 로컬 PyTorch 실행기로 가져와 실행할 수 없어 **미검증**입니다. 성공했다고 주장하지 않습니다.

## 사용 전 CPU 로딩 검사 (데이터 학습/GPU 실험 없음)

Git 저장소 루트에서:

```bash
python dongkyu_work/misa/scripts/verify_prompts.py
```

이 스크립트가 공식 SHA를 다시 계산하고, `torch.load(..., map_location="cpu", weights_only=True)`로 직접 파일을 열어 tensor 여부, dtype, shape를 검사합니다. 성공하면 MISA 모델 `_load_misa_prompts()`에 필요한 **체크포인트 텐서 형식 검증**이 끝납니다. 단, ViT Forward 검증과는 구분해야 합니다.

실행: `python main.py --method misa --misa_prompt_dir ./dongkyu_work/misa/checkpoints ...`

공식 가중치 파일을 임의로 변형하거나 이름만 같은 파일로 대체하지 마십시오.
