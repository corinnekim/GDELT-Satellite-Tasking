"""
Stage-1 모델 비교: 칼만 vs ARIMA vs 튜닝 ML (LightGBM/XGBoost/RandomForest).
같은 GT 매칭(ModelEvaluator)·같은 도시·production 칼만(compute_conflict_index)으로
ROC AUC와 Recall@top-20을 한 표로 낸다. Gemini 호출 없음.
실행: python -m analysis.unified_comparison
"""
from __future__ import annotations
import json
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from pmdarima import auto_arima

from pipeline.config import MAIN_PATH, PROJECT_ROOT
from pipeline.kalman_filter import compute_conflict_index
from analysis.evaluator import ModelEvaluator
from analysis.lightgbm_baseline import build_features, time_split
from analysis.robust_kalman_experiment import load_gt

OUT_DIR = PROJECT_ROOT / "logs" / "unified_comparison"
FIG_DIR = OUT_DIR / "figures"
GT_DIR = PROJECT_ROOT / "data" / "ground_truth"
OUT_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)

TEST_START, TEST_END = '20260322', '20260331'
ARIMA_MIN_NONZERO = 20
TOPK = 20


def fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def roc_auc_safe(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return float('nan')
    fpr, tpr, _ = roc_curve(y, s)
    return float(auc(fpr, tpr))


def bootstrap_metric(y, s, metric_fn, n_boot=1000, seed=0):
    y = np.asarray(y); s = np.asarray(s)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() < 2:
            continue
        try:
            vals.append(metric_fn(y[idx], s[idx]))
        except Exception:
            continue
    if not vals:
        return [float('nan'), float('nan')]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def recall_via_evaluator(scores_df, evaluator, test_dates, k=TOPK):
    """날짜별 top-K 도시를 evaluator로 평가해 Recall."""
    total_hit, total_gt = 0, 0
    per_date = []
    for d in test_dates:
        day = scores_df[scores_df['date_str'] == d].copy()
        if day.empty:
            continue
        day = day.sort_values('score', ascending=False).head(k)
        day = day.assign(llm_status='SUCCESS', std_innov=day['score'],
                         ActionGeo_FeatureID=day.get('feature_id', np.nan))
        m = evaluator.evaluate(day, d)
        total_hit += m['n_gt_hit']; total_gt += m['n_gt']
        per_date.append({'date': d, 'recall': m['recall'],
                         'n_gt': m['n_gt'], 'n_gt_hit': m['n_gt_hit']})
    return {'micro_recall': total_hit / total_gt if total_gt else 0.0,
            'total_hit': total_hit, 'total_gt': total_gt, 'per_date': per_date}


def score_block(name, score_col, evalw, cd_meta, evaluator, test_dates):
    """evalw 위에서 한 모델의 ROC/PR AUC와 Recall 계산. 미적합 도시는 최저 sentinel."""
    sub = evalw[['date_str', 'city', 'target', score_col]].copy()
    coverage = float(sub[score_col].notna().mean())
    if sub[score_col].isna().any():
        sentinel = float(np.nanmin(sub[score_col])) - 1.0
        sub[score_col] = sub[score_col].fillna(sentinel)
    y = sub['target'].values.astype(int)
    s = sub[score_col].values.astype(float)
    roc = roc_auc_safe(y, s)
    pr = float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float('nan')
    roc_ci = bootstrap_metric(y, s, roc_auc_safe)
    pr_ci = bootstrap_metric(y, s, average_precision_score)
    scores_df = (sub[['date_str', 'city']].assign(score=s)
                 .merge(cd_meta, on=['date_str', 'city'], how='left'))
    rec = recall_via_evaluator(scores_df, evaluator, test_dates)
    return {'name': name, 'roc_auc': roc, 'roc_ci': roc_ci, 'pr_auc': pr, 'pr_ci': pr_ci,
            'recall_top20': rec['micro_recall'], 'n_hit': rec['total_hit'],
            'n_gt': rec['total_gt'], 'coverage': coverage, 'y': y.tolist(), 's': s.tolist()}


def diversity_stats(scored, score_col, dates, k=TOPK):
    """날짜별 top-k 도시의 다양성: 고유 수, 상시(≥8/10), 인접일 Jaccard, 상위3 슬롯 비중."""
    per_day = {}
    for d in dates:
        day = scored[scored['date_str'] == d]
        per_day[d] = day.sort_values(score_col, ascending=False).head(k)['city'].tolist() if not day.empty else []
    days = sorted(per_day)
    flat = [c for d in days for c in per_day[d]]
    cnt = Counter(flat)
    jacc = []
    for i in range(len(days) - 1):
        a, b = set(per_day[days[i]]), set(per_day[days[i + 1]])
        if a or b:
            jacc.append(len(a & b) / len(a | b))
    return {'unique_cities': len(cnt),
            'always_8of10': sum(1 for _, n in cnt.items() if n >= 8),
            'mean_adjacent_jaccard': float(np.mean(jacc)) if jacc else 0.0,
            'top3_slot_share': float(sum(n for _, n in cnt.most_common(3)) / len(flat)) if flat else 0.0,
            'top5_cities': [c for c, _ in cnt.most_common(5)]}


def main():
    print("[1/6] conflict_index 계산")
    raw = pd.read_parquet(MAIN_PATH)
    city_daily, f = compute_conflict_index(raw, is_train=False)
    print(f"   city_daily rows: {len(city_daily):,}, cities: {city_daily['city'].nunique():,}")

    # 이름→FeatureID alias 맵
    tmp = raw[['ActionGeo_FullName', 'ActionGeo_FeatureID']].dropna().copy()
    tmp['ActionGeo_FullName'] = tmp['ActionGeo_FullName'].astype(str).str.strip().str.lower()
    name_to_fids = {}
    for nm, g in tmp.groupby('ActionGeo_FullName'):
        s = set()
        for x in g['ActionGeo_FeatureID'].unique():
            try: s.add(int(x))
            except (TypeError, ValueError): pass
        name_to_fids[nm] = s

    gt_df_all = load_gt(GT_DIR)
    evaluator = ModelEvaluator.__new__(ModelEvaluator)
    evaluator.gt_df = gt_df_all; evaluator.radius_km = 50.0
    evaluator.beta = 1.5; evaluator.name_to_fids = name_to_fids
    test_dates = sorted(gt_df_all[(gt_df_all['date'] >= TEST_START) &
                                  (gt_df_all['date'] <= TEST_END)]['date'].unique())
    print(f"   test dates: {len(test_dates)}")

    # (date, city)별 좌표·대표 FeatureID
    cd_meta = (f.groupby(['date', 'standard_name'])
               .agg(lat=('ActionGeo_Lat', 'median'), lng=('ActionGeo_Long', 'median'),
                    feature_id=('ActionGeo_FeatureID',
                                lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan))
               .reset_index().rename(columns={'standard_name': 'city', 'date': 'date_str'}))

    print("[2/6] GT 라벨 생성")
    positives = set()
    for d in sorted(gt_df_all['date'].unique()):
        daily_gt = gt_df_all[gt_df_all['date'] == d]
        for _, row in cd_meta[cd_meta['date_str'] == d].iterrows():
            if evaluator._gt_match(row['lat'], row['lng'], row['city'],
                                   row.get('feature_id'), daily_gt) is not None:
                positives.add((d, row['city']))
    print(f"   positives: {len(positives)}")

    # Kalman 점수 = production std_innov
    kal_df = (city_daily[['date', 'city', 'std_innov']]
              .rename(columns={'date': 'date_str', 'std_innov': 'score_kalman'}))
    kal_df = kal_df[(kal_df['date_str'] >= TEST_START) & (kal_df['date_str'] <= TEST_END)]

    print("[3/6] ARIMA 적합")
    # ARIMA 입력: production conflict_index 시계열
    fit_targets = []
    series_by_city = {}
    for city, g in city_daily.groupby('city'):
        g = g.sort_values('date')
        s = g['conflict_index'].values.astype(np.float64)
        series_by_city[city] = (g['date'].values, s)
        if (s > 0).sum() >= ARIMA_MIN_NONZERO:
            fit_targets.append(city)
    print(f"   fittable cities: {len(fit_targets)} / {city_daily['city'].nunique()} "
          f"({100*len(fit_targets)/city_daily['city'].nunique():.1f}%)")
    arima_rows = []; arima_cities = []; n_ok = 0
    for ci, city in enumerate(fit_targets):
        dates_c, s = series_by_city[city]
        try:
            m = auto_arima(np.log1p(s), start_p=0, start_q=0, max_p=3, max_q=3, max_d=2,
                           seasonal=False, stepwise=True, suppress_warnings=True,
                           error_action='ignore', information_criterion='bic')
            resid = np.asarray(m.resid(), dtype=float)
            zr = resid / (resid.std() if resid.std() > 1e-9 else 1.0)
            arima_cities.append(city); n_ok += 1
            for i, ds in enumerate(dates_c):
                if TEST_START <= ds <= TEST_END and i < len(zr) and np.isfinite(zr[i]):
                    arima_rows.append((ds, city, float(zr[i])))
        except Exception:
            continue
        if (ci + 1) % 50 == 0:
            print(f"   ... {ci+1}/{len(fit_targets)} fitted={n_ok}")
    arima_df = pd.DataFrame(arima_rows, columns=['date_str', 'city', 'score_arima'])
    print(f"   ARIMA fitted: {n_ok}/{len(fit_targets)}")

    print("[4/6] ML feature + 튜닝")
    feat_df = build_features(f, positives, static_until_date=TEST_START)
    train, evalw = time_split(feat_df)
    cmean = train.groupby('country_code')['target'].mean().to_dict()
    train['country_te'] = train['country_code'].map(cmean).fillna(0.0)
    evalw['country_te'] = evalw['country_code'].map(cmean).fillna(0.0)
    base_static = ['city_total_signal', 'city_active_days', 'country_te', 'dow']
    base_today = ['log_sources_sum', 'log_sources_mean', 'log_mentions_sum', 'log_mentions_mean',
                  'avg_tone_mean', 'avg_tone_min', 'avg_tone_std',
                  'goldstein_mean', 'goldstein_min', 'event_count', 'n_actors']
    base_ts = []
    for sig in ['conflict_index', 'log_mentions_sum', 'avg_tone_mean', 'goldstein_mean']:
        base_ts += [f'{sig}_lag1', f'{sig}_lag7', f'{sig}_rmean7',
                    f'{sig}_rstd7', f'{sig}_rmean14', f'{sig}_rstd14']
    feats = base_static + base_today + base_ts
    print(f"   {len(feats)} features | train pos={int(train['target'].sum())} eval pos={int(evalw['target'].sum())}")

    evalw = evalw.merge(kal_df, on=['date_str', 'city'], how='left')
    evalw = evalw.merge(arima_df, on=['date_str', 'city'], how='left')

    def tune_ml(factory, grid, name):
        tscv = TimeSeriesSplit(n_splits=5)
        gs = GridSearchCV(factory(), grid, cv=tscv, scoring='roc_auc', n_jobs=2)
        tr = train.sort_values('date_str').reset_index(drop=True)
        gs.fit(tr[feats].values, tr['target'].values)
        best = gs.best_estimator_
        print(f"   [{name}] best={gs.best_params_}")
        return best.predict_proba(evalw[feats].values)[:, 1], best

    lgb_proba, lgb_est = tune_ml(lambda: lgb.LGBMClassifier(objective='binary', verbose=-1, random_state=0,
                        feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5),
                        {'num_leaves':[15,31,63], 'learning_rate':[0.03,0.05,0.1],
                         'n_estimators':[200,400], 'min_child_samples':[10,20]}, 'LightGBM')
    xgb_proba, _ = tune_ml(lambda: xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc',
                        random_state=0, n_jobs=2, subsample=0.85, colsample_bytree=0.85, tree_method='hist'),
                        {'max_depth':[3,6,9], 'learning_rate':[0.03,0.05,0.1],
                         'n_estimators':[200,400], 'min_child_weight':[1,5]}, 'XGBoost')
    rf_proba, _ = tune_ml(lambda: RandomForestClassifier(random_state=0, n_jobs=2, class_weight='balanced'),
                       {'n_estimators':[200,500], 'max_depth':[5,10,None],
                        'min_samples_leaf':[5,20], 'max_features':['sqrt',0.5]}, 'RandomForest')
    evalw['score_lgb'] = lgb_proba
    evalw['score_xgb'] = xgb_proba
    evalw['score_rf'] = rf_proba

    print("[5/6] 지표 계산")
    blocks = [
        score_block('Kalman', 'score_kalman', evalw, cd_meta, evaluator, test_dates),
        score_block('ARIMA(auto)', 'score_arima', evalw, cd_meta, evaluator, test_dates),
        score_block('LightGBM', 'score_lgb', evalw, cd_meta, evaluator, test_dates),
        score_block('XGBoost', 'score_xgb', evalw, cd_meta, evaluator, test_dates),
        score_block('RandomForest', 'score_rf', evalw, cd_meta, evaluator, test_dates),
    ]

    # 도시별 ARIMA vs Kalman AUC 비교
    pc = []
    for city in arima_cities:
        rows = evalw[(evalw['city'] == city) & evalw['score_arima'].notna()]
        if len(rows) < 5 or rows['target'].sum() < 1:
            continue
        ka = roc_auc_safe(rows['target'].values, rows['score_kalman'].fillna(rows['score_kalman'].min()).values)
        aa = roc_auc_safe(rows['target'].values, rows['score_arima'].values)
        if np.isfinite(ka) and np.isfinite(aa):
            pc.append({'city': city, 'kalman_auc': ka, 'arima_auc': aa})
    pc_df = pd.DataFrame(pc)
    parity = {}
    if len(pc_df):
        kw = int((pc_df['kalman_auc'] > pc_df['arima_auc'] + 0.05).sum())
        aw = int((pc_df['arima_auc'] > pc_df['kalman_auc'] + 0.05).sum())
        parity = {'n_cities': len(pc_df), 'kalman_median': float(pc_df['kalman_auc'].median()),
                  'arima_median': float(pc_df['arima_auc'].median()),
                  'kalman_wins_margin5': kw, 'arima_wins_margin5': aw, 'ties': len(pc_df) - kw - aw}

    print("[6/6] top-20 다양성")
    div = {}
    div['Kalman'] = diversity_stats(evalw, 'score_kalman', test_dates)
    for nm, col in [('LightGBM', 'score_lgb'), ('XGBoost', 'score_xgb'), ('RandomForest', 'score_rf')]:
        div[nm] = diversity_stats(evalw, col, test_dates)
    imp = pd.Series(lgb_est.booster_.feature_importance(importance_type='gain'), index=feats)
    static_gain_share = float(imp[base_static].sum() / imp.sum())

    print("\n" + "=" * 96)
    print(f"[UNIFIED] test {TEST_START}~{TEST_END}")
    print("=" * 96)
    print(f"{'Model':<14} {'ROC AUC':<9} {'95% CI':<18} {'PR AUC':<8} {'Recall@20':<10} {'hit/gt':<9} {'coverage':<9}")
    print("-" * 96)
    for b in blocks:
        hg = f"{b['n_hit']}/{b['n_gt']}"
        print(f"{b['name']:<14} {b['roc_auc']:<9.3f} {fmt_ci(b['roc_ci']):<18} {b['pr_auc']:<8.3f} "
              f"{b['recall_top20']:<10.3f} {hg:<9} {b['coverage']:<9.3f}")
    if parity:
        print("-" * 96)
        print(f"[per-city parity | ARIMA 적합 가능 {parity['n_cities']} 도시] "
              f"Kalman median {parity['kalman_median']:.3f} vs ARIMA {parity['arima_median']:.3f} | "
              f"K {parity['kalman_wins_margin5']} / A {parity['arima_wins_margin5']} / tie {parity['ties']}")
    print("-" * 96)
    print(f"{'[top-20 다양성]':<14} {'uniq':<7} {'always(8/10)':<14} {'adj_jaccard':<13} {'top3_share':<11}")
    for nm in ['Kalman', 'LightGBM', 'XGBoost', 'RandomForest']:
        r = div[nm]
        print(f"{nm:<14} {r['unique_cities']:<7} {r['always_8of10']:<14} "
              f"{r['mean_adjacent_jaccard']:<13.3f} {r['top3_slot_share']:<11.3f}")
    print(f"[LightGBM 정적 feature gain 비중] {static_gain_share:.1%}")
    print(f"[LightGBM 빈출 top5] {div['LightGBM']['top5_cities']}")
    print(f"[Kalman   빈출 top5] {div['Kalman']['top5_cities']}")

    palette = {'Kalman':'#2ecc71','ARIMA(auto)':'#e74c3c','LightGBM':'#3498db',
               'XGBoost':'#e67e22','RandomForest':'#9b59b6'}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    for b in blocks:
        y = np.array(b['y']); s = np.array(b['s'])
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=palette[b['name']], lw=2, label=f"{b['name']} AUC={b['roc_auc']:.3f}")
    ax.plot([0,1],[0,1],'k--',alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.grid(alpha=0.3)
    ax.set_title('ROC curve')
    ax.legend(loc='lower right', fontsize=9)
    ax = axes[1]
    names = [b['name'] for b in blocks]; aucs = [b['roc_auc'] for b in blocks]
    recs = [b['recall_top20'] for b in blocks]
    colors = [palette[n] for n in names]; x = np.arange(len(names)); w = 0.38
    ax.bar(x-w/2, aucs, w, color=colors, alpha=0.85, edgecolor='black', label='ROC AUC')
    ax.bar(x+w/2, recs, w, color=colors, alpha=0.45, edgecolor='black', hatch='//', label='Recall@top-20')
    for i in range(len(blocks)):
        ax.text(x[i]-w/2, aucs[i]+0.01, f'{aucs[i]:.2f}', ha='center', fontsize=8, fontweight='bold')
        ax.text(x[i]+w/2, recs[i]+0.01, f'{recs[i]:.2f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12); ax.set_ylim(0,1.18)
    ax.set_ylabel('Score'); ax.grid(alpha=0.3, axis='y')
    ax.set_title('ROC AUC vs Recall@top-20')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'unified_comparison.png', dpi=120, bbox_inches='tight')
    report_fig = PROJECT_ROOT / 'report' / 'images' / 'model_comparison.png'
    fig.savefig(report_fig, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"\nsaved fig: {FIG_DIR / 'unified_comparison.png'}\n           {report_fig}")

    summary = {
        'test_window': f'{TEST_START}~{TEST_END}',
        'stage1_source': 'pipeline.kalman_filter.compute_conflict_index (production)',
        'universe': 'event city-days in test window (evalw)',
        'matcher': 'ModelEvaluator: name substring -> FeatureID alias -> 50km geo',
        'n_positives_total': len(positives),
        'models': {b['name']: {k: v for k, v in b.items() if k not in ('y', 's')} for b in blocks},
        'arima_parity': parity,
        'topk_diversity': div,
        'lgbm_static_gain_share': static_gain_share,
    }
    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    if len(pc_df):
        pc_df.to_csv(OUT_DIR / 'arima_per_city_auc.csv', index=False)
    print(f"saved: {OUT_DIR / 'summary.json'}")


if __name__ == '__main__':
    main()