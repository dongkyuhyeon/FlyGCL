# AIR — FlyGCL 통합 비교군

공식 코드의 핵심 갱신을 FlyGCL 공통 실행·평가에 연결한 `--method air` 구현입니다.

## 실행

Python 3.10, 저장소 의존성, CUDA GPU 1개와 공통 ViT checkpoint를 준비한 뒤 저장소 루트에서 실행합니다. [공통 환경·평가 조건](../analytic_common/README.md)을 먼저 확인하세요.

```bash
CUDA_VISIBLE_DEVICES=0 DATA_DIR=/path/to/CIFAR bash dongkyu_work/air/scripts/run_air.sh
# 데이터셋·seed 등은 뒤에 추가해 덮어쓸 수 있습니다.
bash dongkyu_work/air/scripts/run_air.sh --seeds 1 2 3 --online_iter 3
```

기본 스크립트: CIFAR100, ViT-B/16, 5 tasks, n=50, m=10, rnd_NM, batch 64, online_iter=1, 1 epoch, autoaug, seen-class 평가. 결과는 기존 FlyGCL 로그·NumPy 경로에 저장됩니다.

## 구현과 공식 코드의 차이

- 공식 `GeneralizedARM`의 클래스별 공분산·합·빈도와 class balancing 수식을 사용합니다. solve로 inverse 연산을 대체했고 공식 결과와 비교합니다.
- 기본 확장 차원 **1024**, gamma **0.1**, float64입니다. 공식 통합 라이브러리 기본값 **8192**와 다릅니다. 100클래스의 `A` 통계만 각각 **0.78125 GiB / 50 GiB**가 필요합니다.
- 공식 차원은 `--air_expansion_size 8192 --air_max_stats_gib 64`로 선택합니다. 작업 메모리가 추가로 필요하며, `--analytic_stats_device cpu` 사용 시에도 충분한 RAM이 필요합니다. 기본 메모리 한도는 8 GiB입니다.
- 별도의 base-session backbone 학습 없이 FlyGCL 공통 사전학습 특징에서 온라인 ARM을 적용합니다. 공식 논문 설정 전체를 그대로 재현했다는 의미는 아닙니다.

## 파일 안내

| 위치 | 내용 |
| --- | --- |
| [implementation/](implementation/) | 실행되는 model · trainer · analytic head |
| [scripts/](scripts/) | 실행 스크립트와 CPU 검증 진입점 |
| [original/](original/) | 선택된 공식 파일 원문, MIT 라이선스, 출처 SHA |
| [notes/VERIFICATION.md](notes/VERIFICATION.md) | 검증 범위·결과·미수행 항목 |

공식 출처: [ZHUANGHP/Analytic-continual-learning](https://github.com/ZHUANGHP/Analytic-continual-learning/tree/0afd6bf87622e808ce7f730f8556e78229b9e758).

```bash
python dongkyu_work/air/scripts/verify.py
```
