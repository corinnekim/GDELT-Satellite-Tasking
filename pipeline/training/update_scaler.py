"""
3개월치 과거 데이터로 StandardScaler를 재학습하고 저장하는 스크립트.
실행: python -m src.update_scaler
"""

import pandas as pd
from pipeline.config import MAIN_PATH, SCALER_PATH
from pipeline.preprocess import apply_standard_scaling

df = pd.read_parquet(MAIN_PATH)

df = df.dropna(subset=['NumMentions', 'NumSources', 'AvgTone', 'GoldsteinScale'])

df = apply_standard_scaling(df, is_train=True)

print(f"스케일러 학습 완료, 샘플 수: {len(df):,}, 저장 경로: {SCALER_PATH}")