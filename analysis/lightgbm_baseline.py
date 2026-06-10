"""
LightGBM baseline — Kalman/ARIMA의 대조군 (제대로 된 feature 설계).

이전 버전의 한계: conflict_index 단일 신호의 lag/rolling만 사용.
- AvgTone, GoldsteinScale, Log_Mentions를 §6.4의 GLM 비유의 근거로 제외했음
- 그러나 LightGBM은 multicollinearity 강건 + 비선형 상호작용 캡처 → 단독 비유의 신호도 함께 쓰면 의미 가능
- 따라서 ML 알고리즘에 공정한 기회를 주기 위해 raw GDELT 신호를 모두 feature로 투입

문제 정의: (city, date) → P(today is GT conflict event) (이진 분류)

평가:
  - ROC AUC, PR AUC + Bootstrap 95% CI
  - Time-based split (train < 2026-02-28, eval 2026-02-28~03-21 같은 ARIMA window)
  - Feature importance (gain) — 어떤 feature가 진짜 기여하는지
  - Ablation: Kalman z-score 유무 비교
"""
from __future__ import annotations
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.model_selection import TimeSeriesSplit

from pipeline.config import MAIN_PATH, PROJECT_ROOT, COEFS
from pipeline.preprocess import clean_gdelt_data, apply_standard_scaling
from pipeline.kalman_filter import kalman_innovation as kalman_gaussian
from pipeline.text_norm import _norm
from analysis.robust_kalman_experiment import compute_a_tier, load_gt

GT_DIR = PROJECT_ROOT / "data" / "ground_truth"
OUT_DIR = PROJECT_ROOT / "logs" / "lightgbm_baseline"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)


def build_features(filtered: pd.DataFrame, positives: set,
                   kalman_std_innov: dict | None = None,
                   static_until_date: str | None = None) -> pd.DataFrame:
    """
    LightGBM용 (city, date) feature 매트릭스.

    Raw GDELT 신호 (도시-일 집계):
      - log_sources_sum, log_mentions_sum    (보도 강도)
      - log_sources_mean, log_mentions_mean
      - avg_tone_mean, avg_tone_min, avg_tone_std   (어조: 평균/최저/변동)
      - goldstein_mean, goldstein_min               (Goldstein: 평균/최저)
      - event_count                                  (도시-일 이벤트 수)
      - n_actors                                     (Actor1 unique 수 — actor 다양성)

    시계열 feature (each of conflict_index, log_mentions_sum, avg_tone_mean, goldstein_mean):
      - lag_1, lag_7
      - rmean_7, rstd_7, rmean_14, rstd_14

    Static:
      - city_total_signal, city_active_days, country_code (target encoded)

    Day:
      - dow

    Optional: kalman_std_innov_t (Kalman 표준화 이노베이션, 시점 t에서 online filter 출력)
              — 미래 정보 사용 없음 (online), Q·R은 §6.5.3 운영 파라미터로 GT-tune 됐음 (이 한계는 누출 논의에서 다룸)
    """
    print("   building base city-day features (raw GDELT signals)...")
    filtered = filtered.copy()
    # raw transformations
    filtered['Log_Sources_raw'] = np.log1p(filtered['NumSources'])
    filtered['Log_Mentions_raw'] = np.log1p(filtered['NumMentions'])

    # city-day aggregation across multiple signals
    cd = (filtered.groupby(['date', 'standard_name'])
          .agg(
              log_sources_sum =('Log_Sources_raw',  'sum'),
              log_sources_mean=('Log_Sources_raw',  'mean'),
              log_mentions_sum =('Log_Mentions_raw', 'sum'),
              log_mentions_mean=('Log_Mentions_raw', 'mean'),
              avg_tone_mean    =('AvgTone',          'mean'),
              avg_tone_min     =('AvgTone',          'min'),
              avg_tone_std     =('AvgTone',          'std'),
              goldstein_mean   =('GoldsteinScale',   'mean'),
              goldstein_min    =('GoldsteinScale',   'min'),
              event_count      =('EventCode',        'count'),
              n_actors         =('Actor1Name',       'nunique'),
              country_code     =('ActionGeo_CountryCode', 'first'),
          )
          .reset_index()
          .rename(columns={'standard_name': 'city'}))
    cd['avg_tone_std'] = cd['avg_tone_std'].fillna(0)
    cd['conflict_index'] = cd['log_sources_sum']  # alias for time-series features

    # static city-level — leakage 방지: static_until_date 이전 데이터만 사용
    # static_until_date 미지정시 전체 기간 사용 (단일 윈도우용 backwards-compat)
    if static_until_date is not None:
        cd_for_static = cd[cd['date'] < static_until_date]
    else:
        cd_for_static = cd
    city_meta = (cd_for_static.groupby('city')
                 .agg(city_total_signal=('conflict_index', 'sum'),
                      city_active_days =('conflict_index', lambda x: (x > 0).sum()))
                 .reset_index())
    cd = cd.merge(city_meta, on='city', how='left')
    # NaN (training data 없는 도시) → 0
    cd['city_total_signal'] = cd['city_total_signal'].fillna(0)
    cd['city_active_days']  = cd['city_active_days'].fillna(0)

    # target encoding for country (mean GT rate per country)
    # done later after target attached

    # full date grid per city
    all_dates = pd.date_range(pd.to_datetime(cd['date']).min(), pd.to_datetime(cd['date']).max(), freq='D')
    cd['date'] = pd.to_datetime(cd['date'])
    cities = sorted(cd['city'].unique())
    print(f"   cities: {len(cities)}, date span {all_dates[0].date()}→{all_dates[-1].date()}")

    rows = []
    for ci, city in enumerate(cities):
        grp = (cd[cd['city'] == city].set_index('date').reindex(all_dates).reset_index()
               .rename(columns={'index': 'date'}))
        grp['city'] = city
        # backfill static
        for col in ['city_total_signal', 'city_active_days', 'country_code']:
            grp[col] = grp[col].ffill().bfill()
        # fill numeric NaN to 0 (no event that day)
        for col in ['log_sources_sum', 'log_sources_mean', 'log_mentions_sum', 'log_mentions_mean',
                    'avg_tone_mean', 'avg_tone_min', 'avg_tone_std',
                    'goldstein_mean', 'goldstein_min', 'event_count', 'n_actors',
                    'conflict_index']:
            grp[col] = grp[col].fillna(0)

        # time-series features for 4 key signals
        for sig in ['conflict_index', 'log_mentions_sum', 'avg_tone_mean', 'goldstein_mean']:
            shifted = grp[sig].shift(1)  # avoid using today's value to predict today
            grp[f'{sig}_lag1'] = shifted.fillna(0)
            grp[f'{sig}_lag7'] = grp[sig].shift(7).fillna(0)
            grp[f'{sig}_rmean7']  = shifted.rolling(7,  min_periods=1).mean().fillna(0)
            grp[f'{sig}_rstd7']   = shifted.rolling(7,  min_periods=1).std().fillna(0)
            grp[f'{sig}_rmean14'] = shifted.rolling(14, min_periods=1).mean().fillna(0)
            grp[f'{sig}_rstd14']  = shifted.rolling(14, min_periods=1).std().fillna(0)

        # day-of-week
        grp['dow'] = grp['date'].dt.dayofweek

        # Kalman 표준화 이노베이션 (optional, hybrid 모델용)
        if kalman_std_innov and city in kalman_std_innov:
            grp['std_innov'] = kalman_std_innov[city]
        else:
            grp['std_innov'] = 0.0

        # target
        grp['date_str'] = grp['date'].dt.strftime('%Y%m%d')
        grp['target'] = grp.apply(lambda r: int((r['date_str'], city) in positives), axis=1)
        rows.append(grp.dropna(subset=['target']))

        if (ci + 1) % 200 == 0:
            print(f"   ... {ci+1}/{len(cities)}")

    df = pd.concat(rows, ignore_index=True)

    # Target encoding for country code (using train mean GT rate)
    # We'll do this after split to avoid leakage. For now keep raw country_code.
    df['country_code'] = df['country_code'].fillna('UNK').astype('category').cat.codes
    print(f"   total rows: {len(df):,}, positives: {df['target'].sum()}")
    return df


def time_split(df: pd.DataFrame,
               train_start='20260228', train_end='20260321',
               eval_start='20260322', eval_end='20260331'):
    """
    보고서 §6.7의 train/test 분할 사용:
      Train: 2026-02-28 ~ 2026-03-21 (22일, GT 220 positives)
      Test:  2026-03-22 ~ 2026-03-31 (10일, GT 100 positives)
    시계열 feature(lag/rolling)는 윈도우 이전 Jan-Feb 데이터를 자동 사용.
    """
    train = df[(df['date_str'] >= train_start) & (df['date_str'] <= train_end)].copy()
    evalw = df[(df['date_str'] >= eval_start)  & (df['date_str'] <= eval_end)].copy()
    return train, evalw


def train_lgb(train_df, feat_cols, params=None):
    if params is None:
        params = {
            'objective': 'binary',
            'metric': ['auc', 'average_precision'],
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'verbose': -1,
        }
    X = train_df[feat_cols].values
    y = train_df['target'].values
    if y.sum() == 0:
        return None
    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain, num_boost_round=400)
    return model


def bootstrap_auc_ci(y, s, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs, prs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], s[idx]
        if yb.sum() < 2: continue
        try:
            aucs.append(float(auc(*roc_curve(yb, sb)[:2])))
            prs.append(float(average_precision_score(yb, sb)))
        except Exception:
            continue
    return ([float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
            [float(np.percentile(prs, 2.5)), float(np.percentile(prs, 97.5))])


def main():
    print("[1/6] Loading + preprocessing data...")
    raw = pd.read_parquet(MAIN_PATH)
    raw['date'] = raw['SQLDATE'].astype(str).str[:8]
    f = clean_gdelt_data(raw)
    f = apply_standard_scaling(f, is_train=False)
    print(f"   filtered rows: {len(f):,}")

    print("\n[2/6] GT positives mapping...")
    gt = load_gt(GT_DIR)
    positives = set()
    cities_set = set(f['standard_name'].unique())
    for _, row in gt.iterrows():
        gn = _norm(str(row['city'])) if pd.notna(row['city']) else ""
        for city in cities_set:
            cn = _norm(str(city).split('|')[0])
            if cn and gn and (cn == gn or cn in gn or gn in cn):
                positives.add((row['date'], city))
    print(f"   positives: {len(positives)} (date, city) pairs")

    print("\n[3/6] Computing Kalman std_innov (online filter, Q=0.001 R=10 per §6.5.3)...")
    f['Log_Sources_raw'] = np.log1p(f['NumSources'])
    agg_k = (f.groupby(['date', 'standard_name'])
             .agg(conflict_index=('Log_Sources_raw', 'sum'))
             .reset_index().rename(columns={'standard_name': 'city'}))
    agg_k['date'] = pd.to_datetime(agg_k['date'])
    all_dates_k = pd.date_range(agg_k['date'].min(), agg_k['date'].max(), freq='D')
    kalman_outputs = {}
    for city in agg_k['city'].unique():
        s = (agg_k[agg_k['city'] == city].set_index('date')['conflict_index']
             .reindex(all_dates_k).fillna(0).values.astype(np.float64))
        if len(s) < 30: continue
        _, _, std_innov = kalman_gaussian(s, 0.001, 10.0)  # §6.5.3 운영 파라미터
        kalman_outputs[city] = std_innov
    print(f"   computed for {len(kalman_outputs)} cities")

    print("\n[4/6] Building rich feature matrix (39 raw GDELT + 1 Kalman std_innov)...")
    df = build_features(f, positives, kalman_outputs)
    train, evalw = time_split(df)
    print(f"\n   train rows: {len(train):,} (pos {train['target'].sum()})")
    print(f"   eval rows:  {len(evalw):,} (pos {evalw['target'].sum()})")

    # Target encoding country_code: mean target on train, applied to both train and eval
    country_mean = train.groupby('country_code')['target'].mean().to_dict()
    train['country_te'] = train['country_code'].map(country_mean).fillna(0.0)
    evalw['country_te'] = evalw['country_code'].map(country_mean).fillna(0.0)

    # Feature lists
    base_static = ['city_total_signal', 'city_active_days', 'country_te', 'dow']
    base_today_signals = ['log_sources_sum', 'log_sources_mean',
                          'log_mentions_sum', 'log_mentions_mean',
                          'avg_tone_mean', 'avg_tone_min', 'avg_tone_std',
                          'goldstein_mean', 'goldstein_min',
                          'event_count', 'n_actors']
    base_ts = []
    for sig in ['conflict_index', 'log_mentions_sum', 'avg_tone_mean', 'goldstein_mean']:
        base_ts.extend([f'{sig}_lag1', f'{sig}_lag7',
                        f'{sig}_rmean7', f'{sig}_rstd7',
                        f'{sig}_rmean14', f'{sig}_rstd14'])
    feats_base = base_static + base_today_signals + base_ts
    feats_hybrid = feats_base + ['std_innov']
    print(f"\n   base features:   {len(feats_base)}  (raw GDELT only)")
    print(f"   hybrid features: {len(feats_hybrid)}  (+ Kalman std_innov)")

    print(f"\n[5/6] Training LightGBM (TimeSeriesSplit 5-fold CV → final test eval)...")
    models = {}
    results = {}
    train_sorted = train.sort_values('date_str').reset_index(drop=True)
    for label, feats in [('base', feats_base), ('hybrid', feats_hybrid)]:
        print(f"\n   === {label} ({len(feats)} features) ===")

        # ── 5-fold TimeSeriesSplit CV within train window ──
        tscv = TimeSeriesSplit(n_splits=5)
        fold_aucs, fold_prs = [], []
        for fi, (tr_idx, va_idx) in enumerate(tscv.split(train_sorted)):
            tr_fold = train_sorted.iloc[tr_idx]
            va_fold = train_sorted.iloc[va_idx]
            if tr_fold['target'].sum() == 0 or va_fold['target'].sum() == 0:
                continue
            m_fold = train_lgb(tr_fold, feats)
            if m_fold is None:
                continue
            sv = m_fold.predict(va_fold[feats].values)
            yv = va_fold['target'].values
            if yv.sum() < 2:
                continue
            fa = float(auc(*roc_curve(yv, sv)[:2]))
            fp = float(average_precision_score(yv, sv))
            fold_aucs.append(fa); fold_prs.append(fp)
            print(f"   Fold {fi}: tr_pos={int(tr_fold['target'].sum())} va_pos={int(yv.sum())}  "
                  f"ROC AUC={fa:.4f}  PR AUC={fp:.4f}")
        cv_auc_mean = float(np.mean(fold_aucs)) if fold_aucs else 0.0
        cv_auc_std  = float(np.std(fold_aucs))  if fold_aucs else 0.0
        cv_pr_mean  = float(np.mean(fold_prs))  if fold_prs  else 0.0
        cv_pr_std   = float(np.std(fold_prs))   if fold_prs  else 0.0
        print(f"   CV: ROC AUC = {cv_auc_mean:.4f} ± {cv_auc_std:.4f}  "
              f"PR AUC = {cv_pr_mean:.4f} ± {cv_pr_std:.4f}  ({len(fold_aucs)} folds)")

        # ── Final: train on full train window, evaluate on held-out test window ──
        m_final = train_lgb(train_sorted, feats)
        if m_final is None:
            print(f"   {label} final train failed"); continue
        y_te = evalw['target'].values
        s_te = m_final.predict(evalw[feats].values)
        roc = float(auc(*roc_curve(y_te, s_te)[:2]))
        pr  = float(average_precision_score(y_te, s_te))
        roc_ci, pr_ci = bootstrap_auc_ci(y_te, s_te)
        print(f"   Test: ROC AUC = {roc:.4f}  95% CI {roc_ci}")
        print(f"   Test: PR  AUC = {pr:.4f}   95% CI {pr_ci}")
        models[label] = (m_final, feats)
        results[label] = {
            'cv_roc_mean': cv_auc_mean, 'cv_roc_std': cv_auc_std,
            'cv_pr_mean': cv_pr_mean, 'cv_pr_std': cv_pr_std,
            'cv_n_folds': len(fold_aucs),
            'test_roc_auc': roc, 'test_pr_auc': pr,
            'test_roc_ci': roc_ci, 'test_pr_ci': pr_ci,
            's_te': s_te.tolist(), 'y_te': y_te.tolist(),
        }

    print(f"\n[6/6] Visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # ROC
    ax = axes[0]
    for label, color in [('base', '#3498db'), ('hybrid', '#9b59b6')]:
        if label not in results: continue
        r = results[label]
        fpr, tpr, _ = roc_curve(np.array(r['y_te']), np.array(r['s_te']))
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"LightGBM {label} ({len(models[label][1])} feat) — "
                      f"test AUC={r['test_roc_auc']:.3f}, CV={r['cv_roc_mean']:.3f}±{r['cv_roc_std']:.3f}")

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_xlim(0, 0.4); ax.set_ylim(0.3, 1.0)
    ax.set_title('LightGBM ROC (test window) — base vs hybrid (with Kalman std_innov)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

    # Feature importance (hybrid model)
    if 'hybrid' in models:
        m, feats = models['hybrid']
        imp = m.feature_importance(importance_type='gain')
        order = np.argsort(imp)[::-1]
        top_n = min(20, len(feats))
        ax = axes[1]
        labels = [feats[i] for i in order[:top_n]][::-1]
        vals = [imp[i] for i in order[:top_n]][::-1]
        colors = ['#9b59b6' if labels[i] == 'std_innov' else 'steelblue' for i in range(len(labels))]
        ax.barh(labels, vals, color=colors)
        ax.set_xlabel('Importance (gain)')
        ax.set_title(f'Feature importance top {top_n} (hybrid model)')

    fig.suptitle('LightGBM: base (39 raw GDELT feat) vs hybrid (+ Kalman std_innov)',
                 fontweight='bold', y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'L1_roc_and_importance.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"   saved: {FIG_DIR / 'L1_roc_and_importance.png'}")

    # Summary
    summary = {k: {kk: vv for kk, vv in v.items() if kk not in ('s_te', 'y_te')}
               for k, v in results.items()}
    summary['n_features'] = {k: len(v[1]) for k, v in models.items()}
    (OUT_DIR / 'lightgbm_results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    print(f"\n   {OUT_DIR / 'lightgbm_results.json'}")


if __name__ == '__main__':
    main()
