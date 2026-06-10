# GDELT 시계열 이상 탐지와 LLM 검증을 활용한 위성 촬영 스케줄링 자동화 대시보드

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?style=flat-square&logo=duckdb&logoColor=black)
![BigQuery](https://img.shields.io/badge/BigQuery-669DF6?style=flat-square&logo=googlebigquery&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-3F51B5?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Kalman Filter](https://img.shields.io/badge/Kalman_Filter-006D77?style=flat-square)
![Gemini API](https://img.shields.io/badge/Gemini_API-8E75B2?style=flat-square&logo=googlegemini&logoColor=white)
![Skyfield](https://img.shields.io/badge/Skyfield-1B2A4A?style=flat-square)
![Next.js](https://img.shields.io/badge/Next.js-000000?style=flat-square&logo=nextdotjs&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white)

분쟁 지역의 군사 사건은 위성 영상으로 평가한다. 폭격 흔적이나 장비 위치 같은 물리적 증거는 은폐·위장·잔해 정리로 빠르게 사라진다. 이 짧은 관측 가능 구간(골든타임) 안에 어떤 도시를 어떤 위성으로 언제 찍을지 자동으로 산출해 판독관의 촬영 의사결정을 돕는 시스템이다.

전 세계 뉴스 이벤트(GDELT)에서 도시별 칼만 필터로 일별 후보 도시를 찾고, 상위 후보의 기사 본문을 LLM이 읽어 위성 촬영 가치가 있는 물리적 분쟁인지 검증한다. 통과한 도시는 위성 궤도·기상과 결합해 촬영 후보를 뽑아 판독관 대시보드로 보여준다. 이란-미국 분쟁 시나리오로 32일 백테스트했고, 수동으로 약 5시간 걸리던 위성 촬영 의사결정을 15분 이내로 단축했다.

<a href="https://sweltering-crane-02b.notion.site/GDELT-LLM-2ec96d4e0a4983eab5ee818b605b3400" target="_blank" rel="noopener noreferrer">Notion 포트폴리오</a> · <a href="https://drive.google.com/file/d/1D5JI0_LX6yRbazqoyO4ce3ORcImtR2Uj/view?usp=sharing" target="_blank" rel="noopener noreferrer">상세 리포트</a> · <a href="https://satellite-dashboard-h875.vercel.app/" target="_blank" rel="noopener noreferrer">대시보드</a>

![판독관 대시보드](report/images/dashboard_overview.png)

좌측 촬영 타임라인과 도시별 위성 스케줄, 중앙 위험도 지도, 우측 선택 도시 상세 분석.

## 문제

기존 방식은 판독관이 뉴스를 수동으로 모니터링해 위기 도시를 찾고, 좌표·기상·위성 궤도를 각각 다른 채널에서 확인한 뒤 촬영을 지시한다. 이 과정에 수 시간이 걸려 골든타임을 놓치는 일이 생긴다. 골든타임을 놓치는 이유는 위성 자원이 부족해서가 아니라 사건 인지에서 촬영 명령까지가 한 시스템으로 통합돼 있지 않기 때문이다. 이 시스템은 **사건·지리·기상·위성 궤도** 네 채널을 단일 파이프라인으로 통합해 촬영 후보를 자동 산출한다.

## 데이터

전 세계 뉴스를 규칙 기반 NLP로 사건화한 공개 데이터셋 [GDELT](https://www.gdeltproject.org) v1.0을 사용했다. 중동 12개국, 2026.02.28~03.31의 32일 약 271만 행이다. 모델은 평시 수준 파악을 위해 전쟁 발발 이전까지 90일을 학습하고, 평가는 전쟁이 시작된 32일 구간에서 한다. GDELT는 한 사건을 여러 행으로 쪼개 기록하며 행위자가 뒤바뀌고 좌표가 어긋나는 등 노이즈가 많다.

## 파이프라인

GDELT는 오류가 많고 희소하며 맥락이 부족하다. 단일 모델로 풀지 않고 단계별로 나눠 각 단계가 한 종류의 노이즈만 책임지게 했다.

`전처리 → 지명 표준화 → 칼만 이상 탐지 → LLM 본문 검증 → 위성 스케줄링 → 판독관 대시보드`

| 단계 | 내용 |
|---|---|
| 전처리 | BigQuery로 GDELT v1.0을 받아 GCS에 parquet 저장. DuckDB SQL로 271만 행을 메모리에 한 번에 올리지 않고 정제(2.1GB→1.2GB). 물리적 분쟁 27개 CAMEO 코드(15x~20x)만 신호로 사용해 Recall 0.33→0.60 |
| 지명 표준화 | 한 도시가 여러 이름·좌표로 흩어진 것을 GeoNames로 한 도시에 통합(신호 합산), 동명 도시는 국가·행정구역으로 분리, 50km 이상 어긋난 좌표 17곳 보정 |
| 칼만 이상 탐지 | 도시별 평시 베이스라인을 학습해 정규화 이노베이션으로 평소 대비 이탈을 측정. 1,123개 도시를 단일 척도로 비교. 라벨 없이 작동하고 매일 수 초에 처리. 관측값 conflict_index는 `log(1+NumSources)` |
| LLM 검증 | 후보 도시의 기사 본문(흩어진 SOURCEURL 풀에서 도시명이 실제 등장하는 기사)을 Gemini Flash-lite가 읽어 물리적 분쟁 여부를 5개 라벨로 판별. 앞 3문장+도시 언급 앞뒤만 잘라 환각·토큰 절감 |
| 위성 스케줄링 | Skyfield로 위성 95대 궤도를 계산해 통과 시각 산출. off-nadir 통과 제외, Astral로 야간 제외, Open-Meteo로 구름량 확인(광학), ICEYE(SAR)는 주야·구름 무관. 도시당 7일 내 가장 빠른 통과 3건 |
| 대시보드 | 판독관이 촬영 대상과 위성 스케줄을 확인하는 3분할 화면(지도·도시 상세·촬영 타임라인) |

## 결과 (32일 백테스트)

정답(Ground Truth)은 GDELT와 독립적인 Wikipedia Current Events와 ISW 보고서에서 위성으로 가시적 변화가 남는 분쟁 사건만 라벨링했다. 위성 산업에서는 잘못 찍는 것보다 찍어야 할 곳을 놓치지 않는 것이 중요하므로 **Recall을 핵심 지표**로 본다.

| 지표 | Train (22일) | Test (10일) |
|---|---|---|
| Recall (A tier, 주요 분쟁 거점) | **0.763** | **0.732** |
| Recall (C tier, 소규모 거점) | 0.321 | 0.267 |
| LLM Precision | 0.775 | 0.765 |

주요 분쟁 거점에서 안정적으로 작동했다(Train/Test 차이 A tier 3.1%p). C tier가 낮은 건 GDELT가 비영어권 소규모 거점을 충분히 잡지 못하기 때문이며, 같은 한계로 전체 Recall 이론 상한은 약 0.87이다.

## 구조

```
pipeline/         운영 파이프라인 (전처리·칼만·LLM·위성·대시보드 JSON)
pipeline/training/  오프라인 학습 (지명 매핑·계수·스케일러·GT 정규화)
analysis/         모델 비교·평가 스크립트 (Kalman vs ARIMA vs ML)
report/           기술 리포트 (단계별 markdown + 이미지)
dashboard/        판독관 대시보드 (Next.js, TypeScript)
data/lookups/     지명 매핑·좌표 보정 테이블
preprocess.ipynb  전처리 (BigQuery→DuckDB 정제·CAMEO 필터·parquet 저장)
eda.ipynb         데이터 EDA (노이즈 구조·변수 분포·conflict_index 계수 결정)
```

운영 파이프라인은 단계별 모듈로 나뉜다.

| 단계 | 모듈 |
|---|---|
| 전처리 | `preprocess.py` · `gdelt_fetcher.py` · `event_dedup.py` |
| 지명 표준화 | `text_norm.py` · `city_utils.py` · `roi_cities.py` |
| 칼만 이상 탐지 | `kalman_filter.py` |
| LLM 검증 | `llm_verification.py` · `cache.py` |
| 위성 스케줄링 | `satellite_catalog.py` · `tle_fetcher.py` · `pass_predictor.py` · `weather_checker.py` · `schedule_builder.py` |
| 대시보드 출력 | `daily_json_writer.py` |
| 오케스트레이션 | `run_pipeline.py` · `config.py` |

## 기술 스택

Python · pandas · Polars · DuckDB · BigQuery · statsmodels · scikit-learn · Trafilatura · Gemini API · Skyfield · Astral · Open-Meteo · Next.js · TypeScript

## 실행

GDELT 원본 데이터와 Gemini·기상 API 키(`GEMINI_API_KEY` 등 환경변수)가 필요하다.

```bash
python -m pipeline.run_pipeline --date 20260228 --mode backtest --save
```
