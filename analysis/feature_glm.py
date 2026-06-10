"""다변량 로지스틱 회귀 — Kalman input feature 선정 정밀 분석.

조건:
  - GT: 22일 train window (2/28~3/21), 188 positives, D-tier 제외
  - GT 매핑은 항상 CONFIRMED_CODES 필터 적용된 GDELT 기준 (188 고정)
  - 후보 feature 5개:
      Log_Sources, Log_Mentions, AvgTone, AvgTone_Sq, GoldsteinScale
  - StandardScaler 적용 (event-level)
  - 두 가지 모드:
      [A] event-level (각 이벤트 = 한 sample, inv-volume weighted)
      [B] city_day-level (도시-날 합 → 한 sample, Kalman 실제 입력 단위)
  - 두 가지 데이터 범위:
      [1] CONFIRMED_CODES 필터 ON (현재 production)
      [2] CONFIRMED_CODES 필터 OFF (전체 이벤트 — 평화 포함)

각 조합에 대해:
  - 각 feature 단독 GLM → p-value, AUC, coef
  - 5 feature 다변량 GLM → 각 p-value, coef, AUC
  - VIF (다중공선성)
  - sklearn class_weight='balanced' coef → production COEFS 후보

총 4개 조합 × 두 분석 (단일/다변량) = 정밀 보고.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from pipeline.config import CONFIRMED_CODES, MONITORED_COUNTRIES, CITY_BLACKLIST, MAIN_PATH, PROJECT_ROOT
from pipeline.preprocess import apply_name_map
from pipeline.text_norm import _norm
from pipeline.training.fit_coefs import haversine, GT_DIR

WIN_START, WIN_END = "20260228", "20260321"
RADIUS_KM, GEO_DOM = 30.0, 0.60
FEATURES = ['Log_Sources', 'Log_Mentions', 'AvgTone', 'AvgTone_Sq', 'GoldsteinScale']


def load_gdelt_window(apply_event_filter: bool):
    raw = pd.read_parquet(MAIN_PATH)
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


def get_positives_fixed():
    """GT mapping은 항상 CONFIRMED_CODES 필터된 gdelt 기준 → GT=188 고정."""
    g = load_gdelt_window(apply_event_filter=True)
    name_to_std = {}
    for _, r in g[['ActionGeo_FullName', 'standard_name']].drop_duplicates().iterrows():
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
                near = g[g.ActionGeo_Lat.between(lat - 0.5, lat + 0.5)
                         & g.ActionGeo_Long.between(lon - 0.5, lon + 0.5)].copy()
                if not near.empty:
                    near['dist'] = haversine(lat, lon, near.ActionGeo_Lat.values, near.ActionGeo_Long.values)
                    near = near[near.dist <= RADIUS_KM]
                if not near.empty:
                    tot = near.groupby('standard_name')['NumMentions'].sum().sort_values(ascending=False)
                    if tot.iloc[0] / tot.sum() >= GEO_DOM:
                        std = tot.index[0]; n_geo += 1
            if std is None:
                n_drop += 1; continue
            positives.add((date, std))
    return positives, (n_name, n_geo, n_drop)


def add_features(g):
    g = g.copy()
    g['Log_Mentions'] = np.log1p(g['NumMentions'])
    g['Log_Sources'] = np.log1p(g['NumSources'])
    g['AvgTone_Sq'] = g['AvgTone'] ** 2
    return g


def build_samples(g, mode, positives):
    """mode = 'event' or 'city_day'."""
    keys = list(zip(g['date'], g['standard_name']))
    if mode == 'event':
        g = g.copy()
        g['y'] = [1 if k in positives else 0 for k in keys]
        counts = g.groupby(['date', 'standard_name']).size().rename('n_cd')
        g = g.merge(counts, left_on=['date', 'standard_name'], right_index=True)
        sample_weight = (1.0 / g['n_cd']).values
        X = g[FEATURES].values
        y = g['y'].values
        return X, y, sample_weight
    elif mode == 'city_day':
        agg = g.groupby(['date', 'standard_name'])[FEATURES].sum().reset_index()
        keys = list(zip(agg['date'], agg['standard_name']))
        agg['y'] = [1 if k in positives else 0 for k in keys]
        X = agg[FEATURES].values
        y = agg['y'].values
        return X, y, None
    raise ValueError(mode)


def fit_glm(X, y, sample_weight, label):
    Xc = sm.add_constant(X)
    if sample_weight is not None:
        m = sm.GLM(y, Xc, family=sm.families.Binomial(), freq_weights=sample_weight).fit()
    else:
        m = sm.Logit(y, Xc).fit(disp=0, maxiter=500)
    return m


def vif_table(X):
    """다중공선성: VIF > 10이면 심각, > 5 주의."""
    Xc = sm.add_constant(X)
    return [variance_inflation_factor(Xc, i+1) for i in range(X.shape[1])]


def run_one_combination(filter_on: bool, mode: str, positives: set, scaler: StandardScaler | None = None):
    print(f"\n{'='*78}")
    print(f"[조합] CONFIRMED_CODES 필터 {'ON' if filter_on else 'OFF'}  ·  mode={mode}")
    print(f"{'='*78}")

    g = load_gdelt_window(apply_event_filter=filter_on)
    g = add_features(g)
    print(f"events: {len(g):,}  city-days: {g.groupby(['date','standard_name']).ngroups:,}")
    # event-level standardize first (consistency)
    if scaler is None:
        scaler = StandardScaler().fit(g[FEATURES])
    g[FEATURES] = scaler.transform(g[FEATURES])
    X, y, w = build_samples(g, mode, positives)
    print(f"samples (mode={mode}): n={len(y):,}  positives={int(y.sum()):,}  ratio={y.mean():.4f}")

    # === 단일 feature GLM ===
    print(f"\n[단일 feature]  {'feature':16s} {'coef':>8s} {'p':>10s} {'AUC':>7s}")
    print("─"*78)
    for i, f in enumerate(FEATURES):
        try:
            m = fit_glm(X[:, [i]], y, w, label=f)
            coef = m.params[1]; pval = m.pvalues[1]
            try:
                auc = roc_auc_score(y, m.predict(sm.add_constant(X[:, [i]])))
            except Exception:
                auc = float('nan')
            print(f"  {f:16s} {coef:>+8.4f} {pval:>10.2e} {auc:>7.3f}")
        except Exception as e:
            print(f"  {f:16s} 적합 실패: {e}")

    # === 다변량 GLM ===
    print(f"\n[다변량 (5 feature)]")
    m = fit_glm(X, y, w, label='multi')
    print(m.summary(xname=['intercept'] + FEATURES).tables[1].as_text())

    auc_multi = roc_auc_score(y, m.predict(sm.add_constant(X)))
    print(f"  AUC = {auc_multi:.4f}")

    # === VIF ===
    vifs = vif_table(X)
    print(f"\n[VIF (다중공선성, 10↑ 심각, 5↑ 주의)]")
    for f, v in zip(FEATURES, vifs):
        warn = ' ← 심각' if v > 10 else ' ← 주의' if v > 5 else ''
        print(f"  {f:16s} {v:>7.2f}{warn}")

    # === sklearn balanced (production COEFS 후보) ===
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')
    clf.fit(X, y, sample_weight=w)
    auc_sk = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    print(f"\n[sklearn class_weight='balanced'] AUC={auc_sk:.4f}")
    print(f"  intercept: {clf.intercept_[0]:+.4f}")
    print(f"  {'feature':16s} {'coef':>10s}")
    for f, c in zip(FEATURES, clf.coef_[0]):
        print(f"  {f:16s} {c:>+10.4f}")
    return scaler


def main():
    print("="*78)
    print("[GT 매핑] CONFIRMED_CODES 필터된 gdelt 기준")
    print("="*78)
    positives, (nn, ng, nd) = get_positives_fixed()
    print(f"  name_exact={nn}  geo_dominant={ng}  D-tier dropped={nd}")
    print(f"  GT positives = {len(positives)} (이게 모든 분석의 정답 set)")

    # 4 조합
    scaler = None
    for filter_on in [True, False]:
        for mode in ['event', 'city_day']:
            scaler = run_one_combination(filter_on, mode, positives, scaler=None)


if __name__ == '__main__':
    main()
