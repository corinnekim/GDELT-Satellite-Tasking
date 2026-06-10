"""
SIA 갈등 모니터링 파이프라인 - 설정파일
- 프로젝트 경로 설정
- 데이터 필터링: CAMEO 코드, 모니터링 대상 국가
- 칼만 필터 파라미터
- 갈등 지수 산출 로직: AvgTone 가중치, EventCode 가중치
- 리스크 레벨 임계값 및 대응 가이드
- 지오코딩 블랙리스트
- LLM 설정
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
SPACETRACK_USER = os.getenv("SPACETRACK_USER")
SPACETRACK_PASSWORD = os.getenv("SPACETRACK_PASSWORD")

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "daily"
SCHEDULE_OUTPUT_DIR = PROJECT_ROOT / "schedule_output"
MAIN_PATH = PROJECT_ROOT / "data/parquet/gdelt_main_2026.parquet"
URL_PATH = PROJECT_ROOT / "data/parquet/gdelt_url_2026.parquet"

# Test mode: 환경변수 GDELT_TEST=1이면 산출물 경로를 test/로 스왑.
# 필터 OFF (CONFIRMED_CODES 미적용) 비교 실험용 격리 폴더.
TEST_MODE = os.getenv("GDELT_TEST") == "1"
ARTIFACT_ROOT = PROJECT_ROOT / "test" if TEST_MODE else PROJECT_ROOT

# Ablation: 성능 개선 OFF (좌표 보정/suffix/fallback/신규 BLACKLIST 모두 비활성)
# baseline 비교 측정용. 위성 단계도 skip.
BASELINE_NO_PERF = os.getenv("GDELT_BASELINE_NO_PERF") == "1"
if BASELINE_NO_PERF:
    ARTIFACT_ROOT = PROJECT_ROOT / "baseline"

# 데이터 필터링

# 물리적 충돌과 관련된 CAMEO 코드 (15, 17, 18, 19, 20) 27개
CONFIRMED_CODES = [
150, 152, 154, 1712, 180, 181, 183, 1831, 1832,
1833, 1834, 186, 190, 191, 192, 193, 194,
195, 1951, 1952, 200, 201, 202, 203, 204, 2041, 2042
]

# 주요 모니터링 국가 (핵심 3국 + 대리전 지역 및 주요 이해관계국)
MONITORED_COUNTRIES = [
    'IR', 'IS',                            # 핵심 국가 (이란, 이스라엘)
    'IZ', 'LE', 'SY', 'YM',                # 분쟁 및 대리전 지역 (이라크, 레바논, 시리아, 예멘)
    'AE', 'SA', 'QA', 'KU', 'BA', 'MU'     # 주요 인접국 (UAE, 사우디, 카타르, 쿠웨이트, 바레인, 오만)
]

# 갈등 지수(Z_t) 산출 로직

# 단일 feature: Log_Sources. CONFIRMED_CODES 필터 적용 기준에서 GLM p=1.3e-03로 가장 유의.
# 의미: "도시·날에 몇 개의 독립 매체가 보도했는가" = reporting intensity.
# 표준화 안 함 (raw log1p 사용): 단일 feature에서 센터링하면 저소스 이벤트가 음수가 되어
# city-day sum에서 상쇄됨. raw log1p 그대로 → 모든 이벤트 ≥0 기여 → 단조 누적 시계열.
COEFS = {
    'w_log_sources': 1.0,
}

SCALING_FEATURES = []

# TEST_MODE 오버라이드: CONFIRMED_CODES 필터 OFF 비교실험용
# Log_Mentions (p=0.27), GoldsteinScale (p=0.63) drop; 유의한 3-feature만.
# 계수는 analysis/feature_glm.py 필터-OFF event mode sklearn balanced 결과.
if TEST_MODE:
    COEFS = {
        'w_log_sources': +0.4283,
        'w_avgtone':     -0.3727,
        'w_avgtone_sq':  -0.3572,
    }
    SCALING_FEATURES = ['Log_Sources', 'AvgTone', 'AvgTone_Sq']

# 스케일러 저장 경로 (훈련 시 저장하고 예측 시 불러옴)
SCALER_PATH = ARTIFACT_ROOT / "models" / "standard_scaler.pkl"

# Kalman Filter 파라미터

MIN_HISTORY = 30  # 칼만 필터 안정화를 위한 최소 관측 일수
KALMAN_MIN_VAR = 1.0    # 초기 최소 분산
KALMAN_Q = 0.001        # 프로세스 노이즈 (고정값, grid search 결과)
KALMAN_R = 10.0         # 관측 노이즈 (고정값, grid search 결과)
KALMAN_Q_RATIO = KALMAN_Q   # auto mode용 (manual_q=-1일 때 init_var에 곱함)
KALMAN_R_RATIO = KALMAN_R   # auto mode용 (manual_r=-1일 때 init_var에 곱함)
KALMAN_P0_RATIO = 2.0   # 초기 불확실성 계수

# 리스크 레벨 임계값 및 대응 가이드

# 임계치와 라벨을 순서대로 정의
# 순서 짝 맞춰서 적기
RISK_THRESHOLDS = [5, 2, 0.5]
# 운영자 검토 큐: high-z인데 LLM이 DROPPED/DATE_MISMATCH 한 케이스를 별도 검토 대상으로
REVIEW_Z_THRESHOLD = 5.0
REVIEW_LLM_STATUSES = {'DROPPED', 'DATE_MISMATCH'}
RISK_LEVELS_LIST = [3, 2, 1]
RISK_LABELS_LIST = ['심각', '주의', '관심']
RISK_GUIDES_LIST = [
    '즉시 대응',
    '정밀 분석',
    '모니터링'
]
RISK_EMOJIS_LIST = ['🛑', '🟠', '🟡']


# 지오코딩 블랙리스트 (조직명/무기명/지명 오류)
CITY_BLACKLIST = {
    'Basij',          # 이란 혁명수비대 민병대 (조직명)
    'Shahed',         # 이란 자폭 드론 이름 (무기명)
    'Hezbollah',      # 레바논 무장단체 (조직명)
    'Hamas',          # 팔레스타인 무장단체 (조직명)
    'Kurdistan',      # 지역명이 도시로 잡힘
    'Arabian Peninsula', # 반도 전체가 도시로 잡힘
    'As Iran',          # GDELT 파싱 오류
    'Sepah', # 이란 혁명수비대(IRGC)를 지칭하는 페르시아어 단어
    'Palestinian Red Crescent', # 팔레스타인 적십자사
    'Red Crescent Society', 'Red Cross',  # 적십자사
    'Khaleej Times', # UAE 신문사 이름
    'Ministry of Foreig', 'Ministry Of Foreign Affairs',  # 외교부
    'Gaza', 'West Bank', 'Gaza City', 'Hebron',   # 팔레스타인 관련 (국가 코드가 IS로 잡혀 수동 제외)
    'Gulf News', 'Al Jazeera', 'Reuters', 'AP', 'AFP',  # 뉴스 매체명
    'Simorgh', 'Shahab', 'Qiam', 'Fotros', 'Zolfaghar',  # 이란 미사일 이름
    'Ministry Of Health',     # 보건부
    'Holy Family Church',     # 교회
    'Unrwa',                  # UN 팔레스타인 난민기구
    'Sayyid',                 # 존칭/이름
    'Sheikh Abdullah',        # 사람 이름
    'Al-Haq',                 # 인권 단체명
}

# LLM 설정
LLM_MODELS = [
    "gemini-2.5-flash-lite",          # 백업
    "gemini-3.1-flash-lite-preview",  # 1차: 더 저렴하고 대부분 벤치마크에서 우위 (preview)
]
LLM_TOP_N = 20          # llm 검증 대상 상위 도시 수 (run_experiment.py --top-k 기본값과 일치)
LLM_TOP_K_URLS = 5      # 도시당 초기 검증 기사 수

# llm 프롬프트 템플릿 (현재 로직에서는 사용 안하지만, 향후 활용 가능)
CAMEO_DEFINITION = {
    150: "Military/police power demonstration",
    152: "Increase military alert status",
    154: "Mobilize armed forces",
    1712: "Destroy property",
    180: "Unconventional violence",
    181: "Abduct, hijack, or take hostage",
    183: "Non-military bombing",
    1831: "Suicide bombing",
    1832: "Vehicular bombing",
    1833: "Roadside bombing",
    1834: "Location bombing",
    186: "Assassinate",
    190: "Conventional military force",
    191: "Blockade or restrict movement",
    192: "Occupy territory",
    193: "Fight with small arms and light weapons",
    194: "Fight with artillery and tanks",
    195: "Aerial weapons (General)",
    1951: "Precision-guided missiles",
    1952: "Drones/Remotely piloted weapons",
    200: "Unconventional mass violence",
    201: "Mass expulsion",
    202: "Mass killings",
    203: "Ethnic cleansing",
    204: "WMD use (General)",
    2041: "Chemical/Biological/Radiological weapons",
    2042: "Nuclear weapons"
}

# Level 2a: 위성 촬영 스케줄 설정
# 위성 카탈로그(DEFAULT_SATELLITES)는 pipeline/satellite_catalog.py 로 분리.
# 군집 정의는 data/satellite_data/*.json 참조.

CLOUD_THRESHOLD = 50        # 구름량 50% 초과 시 EO 촬영 부적합
PREDICTION_HOURS = 336      # 14일 검색 (1차 7일 윈도우 + 2차 7일, 1차에 shootable 없으면 마커용)
PRIMARY_WINDOW_HOURS = 168  # 사건 발생 이후 7일 (대시보드 기본 표시 범위)

# 운영 기본 위성 시나리오, coverage = 모든 위성군 (SpaceEye-T + KOMPSAT-7
# + PlanetScope + ICEYE + Sentinel) 사용해 감시 공백 최소화.
OPERATIONAL_SATELLITE_SCENARIO = "coverage"

# Off-Nadir 운영 상한 30°. 위성 카탈로그의 공칭값(35~45°)이 더 크더라도
# 실무 표준에 맞춰 ±30°로 강제 제한 (500km 고도 기준 앙각 ≈ 57.4°↑).
MAX_OPERATIONAL_OFF_NADIR_DEG = 30.0

# GDELT v1.0 publish 시각: 전날 데이터를 다음날 미국 동부시각 06:00에 업데이트.
# EST=UTC-5 / EDT=UTC-4 → 보수적으로 11 UTC 사용 (EST 기준).
# 즉 target_date(YYYYMMDD)의 위성 스케줄은 (target_date + 1일) 11:00 UTC부터 시작해야 함
# (그 전엔 GDELT 데이터가 아직 발표되지 않아 사건 인지 불가).
GDELT_PUBLISH_OFFSET_HOURS = 24 + 11   # +1일 + 11 UTC
MIN_ELEVATION_DEG = 20.0    # 통과 후보 추출용 최소 앙각 (off-nadir 제한이 후필터로 작동)

TLE_CACHE_DIR = PROJECT_ROOT / "data" / "tle"

# ROI 도시 좌표는 pipeline/roi_cities.py 로 분리 (fallback·테스트용).
# 운영 스케줄링은 LLM 검증 도시로 매일 동적 결정.