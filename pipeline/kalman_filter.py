import numpy as np
import pandas as pd
from numba import jit

from pipeline.preprocess import clean_gdelt_data, apply_standard_scaling
from pipeline.config import (
    KALMAN_Q, KALMAN_R, KALMAN_Q_RATIO, KALMAN_R_RATIO, KALMAN_P0_RATIO, KALMAN_MIN_VAR,
    MIN_HISTORY, RISK_THRESHOLDS, RISK_LEVELS_LIST, 
    RISK_LABELS_LIST, RISK_GUIDES_LIST, RISK_EMOJIS_LIST,
    COEFS
)

# 칼만 필터 코어 (Numba JIT)

@jit(nopython=True)
def kalman_innovation(signal: np.ndarray, manual_q=-1.0, manual_r=-1.0) -> tuple:
    """
    수동 설정값(manual_q, manual_r)이 -1.0이면(기본값) 30일 분산 기반 자동 추정을 수행합니다.
    """
    n = len(signal)
                
    # 초기 30일 데이터를 기준으로 노이즈(Q, R) 자동 추정
    init_window = min(30, n)
    init_var = max(np.var(signal[:init_window]), KALMAN_MIN_VAR)
    
    # 수동 설정값이 있으면 사용, 없으면 자동 계산
    Q = manual_q if manual_q > 0 else init_var * KALMAN_Q_RATIO
    R = manual_r if manual_r > 0 else init_var * KALMAN_R_RATIO

    x_est = np.zeros(n)      
    P_est = np.zeros(n)      
    innovation = np.zeros(n) 
    std_innov = np.zeros(n) 

    x_est[0] = signal[0]
    P_est[0] = R * KALMAN_P0_RATIO

    for t in range(1, n):
        x_pred = x_est[t - 1]
        P_pred = P_est[t - 1] + Q

        # 이노베이션 = 관측 - 예측
        innovation[t] = signal[t] - x_pred
        S = P_pred + R
        
        # 정규화 이노베이션
        std_innov[t] = innovation[t] / max(np.sqrt(S), 1e-10)

        K = P_pred / S 
        x_est[t] = x_pred + K * innovation[t]
        P_est[t] = (1 - K) * P_pred

    return x_est, innovation, std_innov


# 도시별 시계열 그룹화 처리

def apply_kalman_group(group, global_min_date, global_max_date, manual_q: float = -1.0, manual_r: float = -1.0):
    """도시(standard_name)별 첫 등장일부터의 시계열 생성 및 필터 적용. manual_q/manual_r > 0이면 Q/R 수동 주입."""
    if group.empty: return None

    # groupby.apply는 grouper 컬럼을 제외하므로 group.name으로 복구
    city_name = group['city'].iloc[0] if 'city' in group.columns else group.name

    group = group.sort_values('date')
    all_dates = pd.date_range(start=global_min_date, end=global_max_date, freq='D')

    # 뼈대 생성 및 숫자 컬럼 0 채우기
    group = group.set_index('date').reindex(all_dates).reset_index().rename(columns={'index': 'date'})
    num_cols = ['conflict_index', 'events', 'mentions', 'articles', 'sources', 'avg_tone']
    group[num_cols] = group[num_cols].fillna(0)

    # 복구
    group['city'] = city_name

    signal = np.ascontiguousarray(group['conflict_index'].values, dtype=np.float64)
    if len(signal) < MIN_HISTORY: return None

    # Core 연산 호출 (Q/R 주입)
    x_est, innov, std_innov = kalman_innovation(signal, manual_q, manual_r)
    
    group['kalman_est'] = x_est
    group['innovation'] = innov
    group['std_innov'] = std_innov

    # 리스크 등급 분류
    z_scores = group['std_innov'].values
    conditions = [z_scores >= t for t in RISK_THRESHOLDS]
    group['risk_level'] = np.select(conditions, RISK_LEVELS_LIST, default=0)
    group['risk_label'] = np.select(conditions, RISK_LABELS_LIST, default='정상')
    group['risk_guide'] = np.select(conditions, RISK_GUIDES_LIST, default='평시 수준 유지')
    group['risk_emoji'] = np.select(conditions, RISK_EMOJIS_LIST, default='🔵')

    group['is_anomaly'] = group['risk_level'] >= 1
    group['date'] = group['date'].dt.strftime('%Y%m%d')
    
    return group


# 메인 파이프라인 함수

def compute_conflict_index(df: pd.DataFrame, is_train: bool = False, manual_q: float = KALMAN_Q, manual_r: float = KALMAN_R) -> tuple:
    """
    is_train=True: 스케일러 학습 및 저장 모드
    is_train=False: 저장된 스케일러 로드 모드
    """
    # 기본 정제 (국가, 코드 필터링 등)
    filtered = clean_gdelt_data(df)
    if filtered.empty: 
        return pd.DataFrame(), filtered

    # 'date' 컬럼 보장 (KeyError 방지)
    if 'date' not in filtered.columns:
        if 'SQLDATE' in filtered.columns:
            filtered['date'] = filtered['SQLDATE'].astype(str).str[:8]
        else:
            raise KeyError("데이터에 'date' 또는 'SQLDATE' 컬럼이 존재하지 않습니다.")

    # 파생 변수(Log_Mentions·Log_Sources·AvgTone_Sq) 생성. SCALING_FEATURES가 비면 스케일링 생략
    filtered = apply_standard_scaling(filtered, is_train=is_train)
    
    # 가중치 공식: COEFS에 정의된 feature만 선형 결합
    key_to_col = {
        'w_log_mentions': 'Log_Mentions', 'w_log_sources': 'Log_Sources',
        'w_avgtone': 'AvgTone', 'w_avgtone_sq': 'AvgTone_Sq',
        'w_goldstein': 'GoldsteinScale',
    }
    filtered['weighted_index'] = 0.0
    for key, w in COEFS.items():
        filtered['weighted_index'] += w * filtered[key_to_col[key]]

    # 도시별 집계 (standard_name 기준)
    agg = (filtered.groupby(['date', 'standard_name'])
            .agg(
                conflict_index=('weighted_index', 'sum'),
                events=('EventCode', 'count'),
                mentions=('NumMentions', 'mean'),
                articles=('NumArticles', 'mean'),
                sources=('NumSources', 'sum'),
                avg_tone=('AvgTone', 'mean'),
                avg_goldstein=('GoldsteinScale', 'mean'),
                lat=('ActionGeo_Lat', 'median'),
                lng=('ActionGeo_Long', 'median'),
            ).reset_index().rename(columns={'standard_name': 'city'}))

    # 날짜 형식 변환 및 범위 확정
    agg['date'] = pd.to_datetime(agg['date'])
    global_min_date = agg['date'].min()
    global_max_date = agg['date'].max()

    # 도시(standard_name)별 그룹화 및 칼만 필터 호출
    results_df = agg.groupby('city', group_keys=False).apply(
        lambda x: apply_kalman_group(x, global_min_date, global_max_date, manual_q, manual_r)
    )
    
    if results_df is None or results_df.empty: 
        return pd.DataFrame(), filtered

    # 최종 결과 반환 (결측치 제거 및 인덱스 초기화)
    return results_df.dropna(subset=['std_innov']).reset_index(drop=True), filtered


def detect_anomalies(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    if df.empty: return df
    return df[(df['date'] == str(target_date)) & (df['is_anomaly'] == True)].copy()