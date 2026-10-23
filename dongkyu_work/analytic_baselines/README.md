# AIR · GACL · SLDA 통합 사전 준비

검토일: 2026-10-23. 기준 저장소: `dongkyuhyeon/FlyGCL`, commit `73e4d97c95ef4498b729ece319d2f39c89c2626b`.

**세 방법 모두 FlyGCL에 통합 가능한 구조이며, 구현을 시작할 공식 소스와 핵심 계산 검증을 준비했다. 이 브랜치는 사전 준비 단계이며 `--method air/gacl/slda` 실행 기능을 추가한 통합 완료본은 아니다.**

## 대상과 판단

| 방법 | 논문 / 공식 소스 | 통합 판단 | 주요 연결 작업 |
| --- | --- | --- | --- |
| AIR | [Analytic Imbalance Rectifier for Continual Learning](https://arxiv.org/abs/2408.10349), arXiv 최초 공개 2024. [개별 저장소](https://github.com/fang-d/AIR)가 안내하는 [저자 통합 저장소](https://github.com/ZHUANGHP/Analytic-continual-learning) | 가능. 세 방법 중 메모리 대응이 가장 큼 | `GeneralizedARM`의 클래스별 통계 누적·재가중 계산, 평가 직전 `update()`, 공통 백본 연결 |
| GACL | [Exemplar-Free Generalized Analytic Continual Learning](https://papers.nips.cc/paper_files/paper/2024/hash/9713d53ee4f31781304b1ca43266f8d1-Abstract-Conference.html), NeurIPS 2024. [공식 코드](https://github.com/CHEN-YIZHU/GACL) | 가능. 기존 Si-Blurry 코드가 있어 연결 구조가 가까움 | 고정 DeiT-small을 FlyGCL 백본 선택으로 교체, 재귀식 유지, Trainer 반환값·평가 연결 |
| SLDA | [Lifelong Machine Learning with Deep Streaming Linear Discriminant Analysis](https://arxiv.org/abs/1909.01520), CVPR Workshops 2020. [원저자 코드](https://github.com/tyler-hayes/Deep_SLDA) | 가능. 분류기 구조가 비교적 단순함 | 샘플별 평균·공분산 갱신, device 처리, 공통 seen-class 평가 연결 |

SLDA는 저장소의 기존 `slca`나 `sdlora`와 다른 방법이다. 현재 기준 commit의 METHODS에는 위 세 방법이 등록되어 있지 않다.

## 이번에 준비한 것

- `original/`: 출처·라이선스를 유지한 공식 참고 파일 17개. 전체 저장소 사본이 아니라 통합에 필요한 선택 파일이다.
- [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json): 저장소, 정확한 commit, 원본 경로, Git blob SHA.
- [INTEGRATION_PLAN.md](INTEGRATION_PLAN.md): 수정 파일, 공통 프로토콜, 방법별 보존 항목, 통합 검증 기준.
- [scripts/verify_reference_cores.py](scripts/verify_reference_cores.py): 공식 코드의 작은 CPU 수치 검증. 원본 파일 hash도 확인한다.
- [scripts/estimate_memory.py](scripts/estimate_memory.py): 클래스 수와 확장 차원에 따른 통계 행렬 메모리 계산.
- [VERIFICATION.md](VERIFICATION.md): 실제 실행 결과와 검증 범위.

## 바로 복사하면 안 되는 부분

1. **백본:** GACL의 `VITACIL`은 `deit_small_patch16_224`와 384차원에 고정되어 있다. 공통 비교용 통합에서는 FlyGCL의 `--backbone`과 기존 pretrained 로더를 사용한다. 기본 ViT-B/16에서는 특징 차원을 768로 읽어야 한다. 이는 공통 백본을 사용한 방법 비교이며 원 논문의 전체 실험 설정 재현을 의미하지 않는다.
2. **학습 경로:** 세 분류기는 gradient-free 통계 갱신을 한다. 일반 Trainer의 optimizer, scheduler, DDP 경로를 그대로 물려받으면 적절하지 않다. 전용 Trainer에서 통계 갱신과 `(loss, acc)` 반환을 연결한다.
3. **평가:** FlyGCL의 노출 클래스 매핑·seen-class mask·top-k·평가 시점·결과 형식을 사용한다. AIR의 가중치 계산과 SLDA의 역공분산 갱신은 평가 전에 최신 상태가 되도록 한다.
4. **SLDA 초기화:** GACL에 복사된 SLDA는 `first_time`이 task 종료 때 바뀌므로 첫 task의 매 배치에서 `fit_base()`를 호출한다. 이번 준비에서는 이 흐름을 확인했고, 원저자의 샘플별 `StreamingLDA.fit()`을 이식 기준으로 선택했다. 초기화 정책은 구현과 문서에 명시해야 한다.
5. **AIR 메모리:** 공개 구현의 `A`는 `[노출 클래스 수, 확장 차원, 확장 차원]`이며 float64이다. 100개 클래스·기본 차원 8,192이면 A만 **50 GiB**이다. 확장 차원 5,000이면 **18.63 GiB**이다. 백본·다른 통계·중간 계산 공간은 별도이므로 이 숫자는 최대 메모리 측정값이 아니다. `update()`에도 큰 임시 텐서가 생긴다.

AIR의 `buffer_size`는 과거 이미지를 저장하는 replay buffer 크기가 아니라 **랜덤 특징 확장 차원**이다. 이 값을 낮추면 메모리가 줄지만 방법의 하이퍼파라미터가 바뀌므로 설정에 남겨야 한다. 공식 수식을 유지하면서 클래스별 합산의 임시 메모리를 줄이는 방법과 CPU 통계 저장 옵션을 먼저 검토할 수 있다.

## 확인 명령

저장소 루트에서, PyTorch가 설치된 Python으로 실행한다. 데이터와 pretrained 백본은 필요 없다.

```bash
python dongkyu_work/analytic_baselines/scripts/verify_reference_cores.py
python dongkyu_work/analytic_baselines/scripts/estimate_memory.py --classes 100 --expansion 8192
```

현재 준비로 확인한 것은 **공식 구현 확보, 이식 가능성, 핵심 수치 계산**이다. FlyGCL 실행 연결, 실제 백본 입력, GPU 전체 경로, 성능 실험은 아직 수행하지 않았다. 실제 구현 순서는 SLDA → GACL → AIR를 권장한다. 공통 통계형 Trainer 연결을 먼저 검증하고 AIR의 메모리 작업을 이어갈 수 있기 때문이다.
