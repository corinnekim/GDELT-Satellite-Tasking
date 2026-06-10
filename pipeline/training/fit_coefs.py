"""
CITY-DAY 단위 다변량 로지스틱 회귀로 COEFS를 다시 적합한다
(kalman_filter에서 weighted_index→conflict_index가 집계되는 단위와 동일).

production과 동일한 절차:
  1) 이벤트 단위 feature 계산: Log_Mentions, Log_Sources, AvgTone, AvgTone_Sq, GoldsteinScale
  2) 이벤트 단위 feature에 StandardScaler.fit_transform (apply_standard_scaling과 동일)
  3) (date, standard_name)별 scaled feature 합산  ← conflict_index = sum(weighted_index)
  4) (date, standard_name)이 GT positive면 1, 아니면 0
  5) 로지스틱 회귀 (class_weight='balanced') → 계수가 새 COEFS

GT 매핑 (ac_threshold_analysis): name_exact → 30km 내 geo_dominant ≥60% → drop (D-tier).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from pipeline.config import CONFIRMED_CODES, MONITORED_COUNTRIES, CITY_BLACKLIST, PROJECT_ROOT, MAIN_PATH
from pipeline.preprocess import apply_name_map
from pipeline.text_norm import _norm

GDELT = MAIN_PATH
GT_DIR = PROJECT_ROOT / "data" / "ground_truth" / "train"

WIN_START, WIN_END = "20260228", "20260321"
RADIUS_KM, GEO_DOM = 30.0, 0.60
# CLI --feats로 지정; 기본 = Sources+Mentions
DEFAULT_FEATURES = ['Log_Mentions', 'Log_Sources', 'AvgTone', 'AvgTone_Sq', 'GoldsteinScale']
# 집계 단위: 'event' = 행 단위, 'city_day' = (date, standard_name)별 합산
DEFAULT_MODE = 'event'


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (0.5 - np.cos((lat2 - lat1) * p) / 2
         + np.cos(lat1 * p) * np.cos(lat2 * p)
         * (1 - np.cos((lon2 - lon1) * p)) / 2)
    return 12742 * np.arcsin(np.sqrt(a))


def load_window_gdelt(apply_event_filter: bool = True):
    raw = pd.read_parquet(GDELT)
    if apply_event_filter:
        raw = raw[raw.EventCode.astype(int).isin(CONFIRMED_CODES)]
    raw = raw[raw.ActionGeo_CountryCode.isin(MONITORED_COUNTRIES)]
    raw = raw[raw.ActionGeo_Type == 4]
    raw = raw.dropna(subset=['ActionGeo_FullName', 'ActionGeo_Lat', 'ActionGeo_Long'])
    raw['date'] = raw['SQLDATE'].astype(str).str[:8]
    raw = raw[(raw.date >= WIN_START) & (raw.date <= WIN_END)]
    raw = raw[~raw['ActionGeo_FullName'].isin(CITY_BLACKLIST)]
    raw = apply_name_map(raw)
    raw = raw[~raw['standard_name'].isin(CITY_BLACKLIST)]
    return raw


def map_gt_to_positives(gdelt):
    name_to_std = {}
    for _, r in gdelt[['ActionGeo_FullName', 'standard_name']].drop_duplicates().iterrows():
        name_to_std[_norm(r['ActionGeo_FullName'])] = r['standard_name']
        name_to_std[_norm(r['standard_name'])] = r['standard_name']
    name_to_std.pop('', None)

    positives = set()
    n_name = n_geo = n_drop = 0
    for fp in sorted(GT_DIR.glob('*.csv')):
        date = fp.stem
        gt = pd.read_csv(fp, skipinitialspace=True)
        gt = gt.rename(columns={'ActionGeo_FullName': 'city', 'Lat': 'lat', 'Long': 'lon'})
        for _, r in gt.iterrows():
            name, lat, lon = r.get('city'), r.get('lat'), r.get('lon')
            std = name_to_std.get(_norm(name))
            if std is not None:
                n_name += 1
            elif pd.notna(lat) and pd.notna(lon):
                near = gdelt[gdelt.ActionGeo_Lat.between(lat - 0.5, lat + 0.5)
                             & gdelt.ActionGeo_Long.between(lon - 0.5, lon + 0.5)].copy()
                if not near.empty:
                    near['dist'] = haversine(lat, lon, near.ActionGeo_Lat.values, near.ActionGeo_Long.values)
                    near = near[near.dist <= RADIUS_KM]
                if not near.empty:
                    tot = near.groupby('standard_name')['NumMentions'].sum().sort_values(ascending=False)
                    frac = tot.iloc[0] / tot.sum()
                    if frac >= GEO_DOM:
                        std = tot.index[0]; n_geo += 1
            if std is None:
                n_drop += 1
                continue
            positives.add((date, std))
    print(f"  GT mapping: name_exact={n_name}  geo_dominant={n_geo}  D-tier(dropped)={n_drop}")
    print(f"  Unique (date, standard_name) positive pairs: {len(positives)}")
    return positives


def main(features=None, mode=None, apply_event_filter: bool = True):
    features = features or DEFAULT_FEATURES
    mode = mode or DEFAULT_MODE
    print(f"[1/5] Loading GDELT {WIN_START}–{WIN_END}... (mode={mode}, features={features}, event_filter={apply_event_filter})")
    g = load_window_gdelt(apply_event_filter=apply_event_filter)
    # GT 매핑은 production과 동일하게 CONFIRMED_CODES 적용된 gdelt를 쓴다
    g_for_gt = g if apply_event_filter else load_window_gdelt(apply_event_filter=True)
    print(f"      {len(g):,} event rows, {g.date.nunique()} days, {g.standard_name.nunique()} unique standard_names")

    print("[2/5] Event-level feature derivation + standardization...")
    g = g.copy()
    g['Log_Mentions'] = np.log1p(g['NumMentions'])
    g['Log_Sources'] = np.log1p(g['NumSources'])
    g['AvgTone_Sq'] = g['AvgTone'] ** 2
    scaler = StandardScaler()
    g[features] = scaler.fit_transform(g[features])
    print(f"      StandardScaler fit on events:")
    for f, m, s in zip(features, scaler.mean_, scaler.scale_):
        print(f"        {f:15s} mean={m:+.4f}  std={s:.4f}")

    print("[3/5] Mapping GT → (date, standard_name) positives... (using filter=ON gdelt for stability)")
    positives = map_gt_to_positives(g_for_gt)

    print(f"[4/5] Assembling samples at mode='{mode}'...")
    sample_weight = None
    if mode == 'event':
        keys = list(zip(g['date'], g['standard_name']))
        g['y'] = [1 if k in positives else 0 for k in keys]
        # 역-volume 가중: 각 도시-날이 총 가중치 1을 갖게 해
        # Tehran의 300개 이벤트가 Safed의 2개를 압도하지 않게 한다.
        counts = g.groupby(['date', 'standard_name']).size().rename('n_cd')
        g = g.merge(counts, left_on=['date', 'standard_name'], right_index=True)
        sample_weight = (1.0 / g['n_cd']).values
        X = g[features].values
        y = g['y'].values
        w_pos = sample_weight[y == 1].sum()
        w_tot = sample_weight.sum()
        print(f"      event rows: {len(y):,}  positives: {y.sum():,}  ({y.mean():.2%})")
        print(f"      inv-volume weighted: Σw={w_tot:.1f} (=n city-days)  positive Σw={w_pos:.1f} ({w_pos/w_tot:.2%})")
    else:  # city_day 모드
        agg = (g.groupby(['date', 'standard_name'])[features].sum().reset_index())
        keys = list(zip(agg['date'], agg['standard_name']))
        agg['y'] = [1 if k in positives else 0 for k in keys]
        X = agg[features].values
        y = agg['y'].values
        print(f"      city-day rows: {len(y):,}  positives: {y.sum():,}  ({y.mean():.2%})")
    FEATURES = features

    print("[5/5] Logistic regression (statsmodels, for p-values)...")
    Xc = sm.add_constant(X)
    if sample_weight is not None:
        # GLM은 freq_weights로 분수 가중치를 받음 (volume 정규화)
        sm_model = sm.GLM(y, Xc, family=sm.families.Binomial(),
                          freq_weights=sample_weight).fit()
        print("\n=== statsmodels GLM Binomial (inv-volume weighted; for p-value inference) ===")
    else:
        sm_model = sm.Logit(y, Xc).fit(disp=0, maxiter=200)
        print("\n=== statsmodels Logit (unweighted; for p-value inference) ===")
    print(sm_model.summary(xname=['intercept'] + FEATURES).as_text())

    # sklearn: sample_weight 위에 class_weight='balanced'를 곱함.
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')
    clf.fit(X, y, sample_weight=sample_weight)

    print("\n=== sklearn Logit (class_weight='balanced'; used for production COEFS) ===")
    print(f"  intercept: {clf.intercept_[0]:+.4f}")
    print(f"  {'feature':15s}  {'coef':>10s}  {'|coef|':>8s}")
    for f, c in zip(FEATURES, clf.coef_[0]):
        print(f"  {f:15s}  {c:>10.4f}  {abs(c):>8.4f}")

    print("\n=== Proposed COEFS block (drop into config.py) ===")
    key_map = {
        'Log_Mentions':   'w_log_mentions',
        'Log_Sources':    'w_log_sources',
        'AvgTone':        'w_avgtone',
        'AvgTone_Sq':     'w_avgtone_sq',
        'GoldsteinScale': 'w_goldstein',
    }
    print("COEFS = {")
    for f, c in zip(FEATURES, clf.coef_[0]):
        print(f"    '{key_map[f]}': {c:+.4f},")
    print("}")

    p = clf.predict_proba(X)[:, 1]
    print(f"\n=== Train-set sanity ===")
    print(f"  ROC AUC : {roc_auc_score(y, p):.4f}")
    print(f"  PR  AUC : {average_precision_score(y, p):.4f}  (baseline pos rate = {y.mean():.4f})")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', nargs='+', default=DEFAULT_FEATURES)
    ap.add_argument('--mode', choices=['event', 'city_day'], default=DEFAULT_MODE)
    ap.add_argument('--no-event-filter', action='store_true',
                    help='CONFIRMED_CODES 필터 끄고 전체 이벤트로 GLM')
    args = ap.parse_args()
    main(features=args.feats, mode=args.mode, apply_event_filter=not args.no_event_filter)
