"""
SIA GDELT 1.0 이벤트 데이터 수량 모듈
GDELT 1.0 마스터 리스트에서 일별 이벤트 CSV를 다운로드하고,
관심 지역(ROI) 필터링을 적용하여 로컬에 저장합니다.
"""

import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from pipeline.config import (
    DATA_DIR, MAIN_PATH, URL_PATH, CONFIRMED_CODES, MONITORED_COUNTRIES,
)

# GDELT 1.0 마스터 파일 리스트 URL
GDELT_MASTER_URL = "http://data.gdeltproject.org/events/index.html"
GDELT_BASE_URL = "http://data.gdeltproject.org/events/"

# GDELT 1.0 이벤트 주요 컬럼 (58개)
GDELT_COLUMNS = [
    'GLOBALEVENTID', 'SQLDATE', 'MonthYear', 'Year', 'FractionDate',
    'Actor1Code', 'Actor1Name', 'Actor1CountryCode', 'Actor1KnownGroupCode',
    'Actor1EthnicCode', 'Actor1Religion1Code', 'Actor1Religion2Code',
    'Actor1Type1Code', 'Actor1Type2Code', 'Actor1Type3Code',
    'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2KnownGroupCode',
    'Actor2EthnicCode', 'Actor2Religion1Code', 'Actor2Religion2Code',
    'Actor2Type1Code', 'Actor2Type2Code', 'Actor2Type3Code',
    'IsRootEvent', 'EventCode', 'EventBaseCode', 'EventRootCode',
    'QuadClass', 'GoldsteinScale', 'NumMentions', 'NumSources',
    'NumArticles', 'AvgTone',
    'Actor1Geo_Type', 'Actor1Geo_FullName', 'Actor1Geo_CountryCode',
    'Actor1Geo_ADM1Code', 'Actor1Geo_Lat', 'Actor1Geo_Long', 'Actor1Geo_FeatureID',
    'Actor2Geo_Type', 'Actor2Geo_FullName', 'Actor2Geo_CountryCode',
    'Actor2Geo_ADM1Code', 'Actor2Geo_Lat', 'Actor2Geo_Long', 'Actor2Geo_FeatureID',
    'ActionGeo_Type', 'ActionGeo_FullName', 'ActionGeo_CountryCode',
    'ActionGeo_ADM1Code', 'ActionGeo_Lat', 'ActionGeo_Long', 'ActionGeo_FeatureID',
    'DATEADDED', 'SOURCEURL',
]

def apply_monitored_filter(df: pd.DataFrame) -> pd.DataFrame:
    """모니터링 국가, 분쟁 코드, 도시 단위(ActionGeo_Type=4) 필터 적용"""
    mask = (
        (df['Actor1CountryCode'].isin(MONITORED_COUNTRIES) |
         df['Actor2CountryCode'].isin(MONITORED_COUNTRIES)) &
        df['EventCode'].astype(str).str.split('.').str[0].isin(CONFIRMED_CODES) &
        (df['ActionGeo_Type'] == 4)
    )
    return df[mask].copy()

def fetch_daily(target_date: str, save: bool = True) -> pd.DataFrame:
    """특정 날짜의 GDELT v1.0 이벤트를 다운로드하고 필터링하여 수집"""
    print(f"--- GDELT 데이터 수집 시작: {target_date} ---")

    save_path = DATA_DIR / f"{target_date}.parquet"
    if save_path.exists():
        print(f"  [로컬 경로] {save_path.name} 파일이 이미 존재합니다.")
        return pd.read_parquet(save_path)

    # GDELT v1.0: 일별 단일 파일 (YYYYMMDD.export.CSV.zip)
    url = f"{GDELT_BASE_URL}{target_date}.export.CSV.zip"
    print(f"  [다운로드] {url}")

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except requests.exceptions.HTTPError:
        print(f"  [오류] {target_date}에 해당하는 GDELT 파일이 없습니다.")
        return pd.DataFrame()

    dfs = []
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                batch = pd.read_csv(f, sep='\t', header=None, names=GDELT_COLUMNS, 
                                    dtype=str, on_bad_lines='skip')
                
                # 수치형 데이터 변환
                for c in ['NumMentions', 'NumSources', 'NumArticles', 'ActionGeo_Type']:
                    batch[c] = pd.to_numeric(batch[c], errors='coerce').fillna(0).astype(int)
                for c in ['AvgTone', 'GoldsteinScale', 'ActionGeo_Lat', 'ActionGeo_Long']:
                    batch[c] = pd.to_numeric(batch[c], errors='coerce').fillna(0.0)
                
                dfs.append(batch)
    except Exception as e:
        print(f"  [오류] 파일 처리 실패: {e}")
        return pd.DataFrame()

    if not dfs: return pd.DataFrame()

    # 데이터 병합 및 필터링
    all_data = pd.concat(dfs, ignore_index=True)
    filtered = apply_monitored_filter(all_data)
    print(f"  [완료] 총 {len(filtered):,}건의 유효한 이벤트를 선별했습니다.")

    # 저장
    if save and not filtered.empty:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        filtered.to_parquet(save_path, index=False)

    return filtered

def load_all_data(include_daily: bool = True) -> pd.DataFrame:
    """기존 대용량 Parquet 파일과 로컬 일별 파일들을 병합하여 로드"""
    dfs = []
    
    # 히스토리 데이터 로드
    if MAIN_PATH.exists():
        print(f"  [히스토리 요약] {MAIN_PATH.name} 로드 중...")
        dfs.append(pd.read_parquet(MAIN_PATH))

    # 일별 수집 데이터 로드
    if include_daily and DATA_DIR.exists():
        daily_files = sorted(DATA_DIR.glob("*.parquet"))
        if daily_files:
            print(f"  [일별 데이터] {len(daily_files)}개 파일을 병합 중...")
            for f in daily_files: dfs.append(pd.read_parquet(f))

    if not dfs: return pd.DataFrame()

    # 병합 및 중복 제거
    combined = pd.concat(dfs, ignore_index=True)
    if 'GLOBALEVENTID' in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(subset=['GLOBALEVENTID'])
        after = len(combined)
        if before > after:
            print(f"  [중복 제거] {before:,} -> {after:,} 건으로 압축 완료.")

    return combined
