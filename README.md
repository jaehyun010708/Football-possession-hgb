# 축구 점유권 분류 — HGB Baseline

IDSSE 트래킹 데이터로 매 프레임의 점유 상태 **H(Home) / A(Away) / D(Deadball)**를 예측합니다.

- **모델:** HistGradientBoostingClassifier + Markov(Viterbi) 후처리
- **Feature:** 선수 191개 / 선수 191개 + 공 11개 비교
- **데이터:** IDSSE 7경기 (학습 6경기, 검증 `J03WPY`), 5 FPS

## 1. 설치 및 실행

Python **3.14.4** 기준, 프로젝트 최상위 폴더에서 실행합니다.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py
```

`run_pipeline.py` 하나로 **데이터 준비 → Feature 생성 → HGB 학습 → Markov 후처리 → 평가**까지 진행합니다.

## 2. 데이터셋 다운로드

데이터 출처: [PySport IDSSE (Hugging Face)](https://huggingface.co/datasets/pysport/idsse-data)

**자동 다운로드:** `python run_pipeline.py`를 실행하면 필요한 원본 XML을 자동으로 내려받습니다. 데이터는 GitHub에 포함되지 않습니다.

**수동 준비:** 이미 XML 파일이 있다면 `data/external/idsse/` 아래에 경기별 폴더를 만들고 넣어 주세요.

```text
data/external/idsse/
├── J03WMX/          # 경기별로 아래 XML 2개 필요
│   ├── DFL_02_01_matchinformation_*.xml
│   └── DFL_04_03_positions_raw_observed_*.xml
├── J03WN1/
├── J03WOH/
├── J03WOY/
├── J03WPY/          # 검증 경기
├── J03WQQ/
└── J03WR9/
```

파일명은 원래 DFL 파일명을 유지해야 합니다. 나머지 경기 폴더에도 같은 종류의 XML 2개가 필요합니다.

## 3. 파일별 역할

| 파일 / 폴더 | 역할 |
|---|---|
| `run_pipeline.py` | 전체 실험 실행 |
| `src/config.py` | 데이터 경로, 경기 구분, Markov 강도 설정 |
| `src/player_features.py` | 선수 Feature 191개 생성·IDSSE 다운로드 |
| `src/ball_features.py` | 공 Feature 11개 생성 |
| `src/training.py` | 선수 기반 HGB 학습 |
| `src/markov.py` | 전이확률 및 Viterbi 후처리 |
| `src/evaluate.py` | 두 HGB 모델 학습·비교 |
| `src/metrics.py`, `src/official_metric.py` | Spell F1 계산·공식 평가 코드 준비 |
| `experiments/` | 개별 실험 실행 스크립트 |
| `scripts/` | 공식 평가 코드 다운로드·점수 재현 확인 |
| `tests/` | Feature·Markov 단위 테스트 |
| `evaluation/` | 실행 시 공식 평가 코드 저장 |

## 4. 실행 후 생성되는 폴더

```text
data/external/idsse/processed/       # Feature Parquet 캐시
models/idsse_temporal_v1.joblib      # 선수 기반 HGB 모델
outputs/idsse_temporal/              # 선수 모델 예측 결과
outputs/idsse_ball_hgb_v1/
├── comparison.json                  # Raw / Markov 평가 점수
├── experiment_config.json           # 실험 설정
├── J03WPY_comparison_5fps.parquet   # 검증 예측 결과
└── idsse_hgb_ball_model.joblib      # 공 추가 HGB 모델
```

원본 데이터, 캐시, 학습 모델, 출력 결과는 `.gitignore`로 Git 업로드에서 제외합니다.

## 5. 검증 결과 (IDSSE J03WPY)

| 모델 | Macro F1 | Spell F1 |
|---|---:|---:|
| 선수 HGB + Markov | 0.8574 | 0.4505 |
| 선수 + 공 HGB + Markov | **0.9269** | **0.6056** |

위 점수는 **기존 환경의 IDSSE 검증 결과**이며 Barcelona 대회 점수가 아닙니다. Markov 강도는 `0.25`입니다.

```bash
python scripts/check_reference_scores.py   # 실행 결과와 기존 점수 비교
```

