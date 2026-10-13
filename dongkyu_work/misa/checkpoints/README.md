# Official MISA initial-session pretrained prompts

이 디렉터리는 로컬 실행 시 공식 MISA의 다음 가중치를 배치할 장소입니다. **현재 이 Git 저장소에는 .pt 파일이 없습니다.**

- `g_prompt.pt` — https://github.com/kangzhiq/MISA/blob/72e7c960c73eceb1e3f63950d18ad6baff52b560/pretrained_prompt/g_prompt.pt
- `e_prompt.pt` — https://github.com/kangzhiq/MISA/blob/72e7c960c73eceb1e3f63950d18ad6baff52b560/pretrained_prompt/e_prompt.pt

저장소 루트에서 `python main.py --method misa --misa_prompt_dir ./dongkyu_work/misa/checkpoints ...`로 사용합니다.

용량과 원본 관리 때문에 가중치는 Git에서 제외하며, 가중치 무결성/텐서 호환성 및 실제 실행은 아직 검증하지 않았습니다.
