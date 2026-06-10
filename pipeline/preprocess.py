"""
GDELT 데이터 필터링 및 중복 제거
- 국가, 이벤트 코드, 지리 정보 기반 노이즈 필터링
- Actor1Name, Actor2Name 기반 중복 제거
- ActionGeo_FeatureID 기반 대표 지명 통합
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from pipeline.config import MONITORED_COUNTRIES, CONFIRMED_CODES, CITY_BLACKLIST, SCALING_FEATURES, SCALER_PATH, ARTIFACT_ROOT, TEST_MODE, BASELINE_NO_PERF

NAME_MAP_PATH = ARTIFACT_ROOT / "data" / "lookups" / "gdelt_name_map.csv"
LATLON_CORRECTION_PATH = Path(__file__).resolve().parent.parent / "data" / "lookups" / "gdelt_latlon_correction.csv"


def apply_latlon_correction(df: pd.DataFrame) -> pd.DataFrame:
    """GDELT geocoding 오류 수정. (FullName, country) → (true_lat, true_lon)으로 좌표 강제.
    GT 검증 좌표 + ROI 수동 보정. 50km 이상 차이나는 케이스만 등재됨.
    이름 통합과 별개, name_map은 그대로 두고 좌표만 정정한다."""
    if BASELINE_NO_PERF:
        return df  # baseline 모드: 좌표 보정 OFF
    if not LATLON_CORRECTION_PATH.exists() or df.empty:
        return df
    ov = pd.read_csv(LATLON_CORRECTION_PATH)
    key_df = df['ActionGeo_FullName'].astype(str) + '||' + df['ActionGeo_CountryCode'].astype(str)
    key_ov = ov['gdelt_fullname'].astype(str) + '||' + ov['country'].astype(str)
    lat_map = dict(zip(key_ov, ov.true_lat))
    lon_map = dict(zip(key_ov, ov.true_lon))
    mask = key_df.isin(lat_map)
    if mask.any():
        target_dtype = df['ActionGeo_Lat'].dtype
        df.loc[mask, 'ActionGeo_Lat']  = key_df[mask].map(lat_map).astype(target_dtype)
        df.loc[mask, 'ActionGeo_Long'] = key_df[mask].map(lon_map).astype(target_dtype)
    return df


def apply_name_map(df: pd.DataFrame) -> pd.DataFrame:
    """GeoNames 기반 매핑으로 `standard_name` 컬럼을 추가한다.
    칼만은 `standard_name`으로 집계하고, LLM은 기사 매칭을 위해 `ActionGeo_FullName`을
    유지한다 (신호는 Kuwait International Airport → Al Farwānīyah로 합치되,
    LLM은 기사에서 "Kuwait International Airport"를 그대로 검색).
    """
    if not NAME_MAP_PATH.exists():
        df['standard_name'] = df['ActionGeo_FullName']
        return df
    nmap = pd.read_csv(NAME_MAP_PATH)
    key = df['ActionGeo_FullName'].astype(str) + '||' + df['ActionGeo_CountryCode'].astype(str)
    nmap_key = nmap['ActionGeo_FullName'].astype(str) + '||' + nmap['ActionGeo_CountryCode'].astype(str)
    mapping = dict(zip(nmap_key, nmap['standard_name'].astype(str)))
    df['standard_name'] = key.map(mapping).fillna(df['ActionGeo_FullName'])
    if BASELINE_NO_PERF:
        # baseline 모드: country suffix 제거
        df['standard_name'] = df['standard_name'].str.split('|').str[0]
    return df

def unify_actiongeo_names(df: pd.DataFrame) -> pd.DataFrame:
    # Tehran/Teheran 수동 통합 로직 (FeatureID와 이름을 하나로 강제 고정)
    # 'Teheran'이든 'Tehran'이든 상관없이
    is_tehran = df['ActionGeo_FullName'].isin(['Teheran', 'Tehran'])

    if is_tehran.any():
        # ID를 standard_id(보통 Tehran의 ID)로 통일
        df.loc[is_tehran, 'ActionGeo_FeatureID'] = '10074674'   # Tehran의 대표 FeatureID로 통일 (GDELT에서 가장 빈번하게 등장)
        # 이름도 'Tehran'으로 통일
        df.loc[is_tehran, 'ActionGeo_FullName'] = 'Tehran'
    

    # FeatureID 기준으로 가장 자주 등장하는 지명을 대표 이름으로 선정
    name_counts = df.groupby(['ActionGeo_FeatureID', 'ActionGeo_FullName']).size().reset_index(name='count')
    best_names = name_counts.sort_values('count', ascending=False).groupby('ActionGeo_FeatureID').first()
    
    # ID당 지명이 여러 개 묶인 경우를 찾아 로그 출력
    id_name_nunique = df.groupby('ActionGeo_FeatureID')['ActionGeo_FullName'].nunique()
    inconsistent_ids = id_name_nunique[id_name_nunique > 1].index
    
    if len(inconsistent_ids) > 0:
        print("\n[전처리] 지명이 여러 개로 나타나는 데이터를 대표 지명으로 통합하였습니다.")
        for fid in inconsistent_ids:
            merged_names = df[df['ActionGeo_FeatureID'] == fid]['ActionGeo_FullName'].unique()
            primary_name = best_names.loc[fid, 'ActionGeo_FullName']
            print(f"  - ID {fid}: {list(merged_names)} -> [{primary_name}]")
            
    # 원본 데이터프레임에 대표 지명 매핑
    mapping_dict = best_names['ActionGeo_FullName'].to_dict()
    df['ActionGeo_FullName'] = df['ActionGeo_FeatureID'].map(mapping_dict)
    
    return df

def clean_gdelt_data(df: pd.DataFrame, apply_event_filter: bool | None = None) -> pd.DataFrame:
    """apply_event_filter=False면 CONFIRMED_CODES (CAMEO 15/17/18/19/20) 컷 생략.
    Kalman/GLM 입력으로 평화/외교 이벤트까지 포함하고 싶을 때 사용. 국가/지오타입/블랙리스트 컷은 유지.
    apply_event_filter=None (default): TEST_MODE면 False, 아니면 True (production 동작)."""
    if apply_event_filter is None:
        apply_event_filter = not TEST_MODE
    """GDELT 원본 데이터를 정제하고 중복을 제거하여 반환"""
    
    if df.empty:
        return pd.DataFrame()

    # 도메인 기반 필터링 (마스크 적용)
    mask = (
        (df['ActionGeo_CountryCode'].isin(MONITORED_COUNTRIES))  &
        (df['ActionGeo_Type'] == 4) &
        (df['NumSources'] >= 1) &
        (~df['ActionGeo_FullName'].isin(CITY_BLACKLIST))
    )
    if apply_event_filter:
        mask &= df['EventCode'].isin(CONFIRMED_CODES)
    
    filtered = df[mask].copy()
    if filtered.empty:
        return filtered

    # Actor 수 기반 중복 제거 
    filtered['info_count'] = filtered[['Actor1Name', 'Actor2Name']].notna().sum(axis=1)
    filtered = filtered.sort_values(by='info_count', ascending=False)

    filtered = filtered.drop_duplicates(
        subset=[
            'SQLDATE', 'ActionGeo_FeatureID', 'EventCode', 
            'AvgTone', 'NumArticles', 'NumSources'
        ],
        keep='first'
    ).drop(columns=['info_count'])

    # 최종 정제된 데이터에 대해 지명 통합 수행
    filtered = unify_actiongeo_names(filtered)

    # GDELT 좌표 오류 보정 (GT 좌표 + ROI 수동 보정 기반).
    # name_map보다 먼저, name_map은 좌표 변경 없이 standard_name만 부여하므로 순서 무관.
    filtered = apply_latlon_correction(filtered)

    # GeoNames 기반 표준명 부여 (ActionGeo_FullName 원본 보존, standard_name 추가)
    filtered = apply_name_map(filtered)

    # 표준명 기준으로 블랙리스트 재필터 (Hebron 등이 표준 이름으로 부활할 수 있음)
    filtered = filtered[~filtered['standard_name'].isin(CITY_BLACKLIST)]

    return filtered


def apply_standard_scaling(df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
    """학습된 스케일러를 적용하거나, 학습 시 새로 저장.
    SCALING_FEATURES가 비어 있으면 스케일링 생략 (raw log1p 등 단조 변환만 사용하는 경우).
    """
    if df.empty: return df

    # 파생 변수 생성 (가중치 공식에 필요한 컬럼들)
    df['Log_Mentions'] = np.log1p(df['NumMentions'])
    df['Log_Sources'] = np.log1p(df['NumSources'])
    df['AvgTone_Sq'] = df['AvgTone'] ** 2

    # SCALING_FEATURES가 비어 있으면 스케일링 스킵
    if not SCALING_FEATURES:
        return df

    scaler_file = Path(SCALER_PATH)

    if is_train:
        scaler = StandardScaler()
        # 4개 변수(Log_Mentions, AvgTone, AvgTone_Sq, GoldsteinScale)에 대해 스케일링 학습 및 적용
        df[SCALING_FEATURES] = scaler.fit_transform(df[SCALING_FEATURES])

        # 스케일러 파일 저장
        scaler_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_file)
        print(f"[전처리] 새로운 스케일러가 {scaler_file}에 저장되었습니다.")
    else:
        # 저장된 스케일러 로드하여 적용 (평균/표준편차 그대로 사용)
        if not scaler_file.exists():
            raise FileNotFoundError("스케일러 파일이 없습니다. 먼저 학습 데이터를 사용하여 fit을 수행하세요.")
        scaler = joblib.load(scaler_file)
        df[SCALING_FEATURES] = scaler.transform(df[SCALING_FEATURES])
        
    return df