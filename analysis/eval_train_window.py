"""22일 학습 윈도우(2/28~3/21) Precision/Recall 평가.

GT: data/ground_truth/train/{date}.csv
매핑: eval_test_window.py와 동일 (CONFIRMED_CODES + CITY_BLACKLIST + name_map +
substring + geo_dominant 50km/60% + suffix strip)
"""
import json, pandas as pd
from pathlib import Path
from pipeline.config import MAIN_PATH, MONITORED_COUNTRIES, CITY_BLACKLIST, CONFIRMED_CODES, PROJECT_ROOT
from pipeline.preprocess import apply_name_map
from pipeline.text_norm import _norm
from pipeline.training.fit_coefs import haversine

DATES = ([f"202602{d:02d}" for d in (28,)] +
         [f"202603{d:02d}" for d in range(1, 22)])
TRAIN_GT = PROJECT_ROOT / "data" / "ground_truth" / "train"


def gt_for_date(d, raw):
    sub = raw[raw['date'] == d]
    if sub.empty:
        return set()
    n2s = {}
    for _, r in sub[['ActionGeo_FullName', 'standard_name']].drop_duplicates().iterrows():
        n2s[_norm(r['ActionGeo_FullName'])] = r['standard_name']
        n2s[_norm(r['standard_name'])] = r['standard_name']
    gt = pd.read_csv(TRAIN_GT / f"{d}.csv", skipinitialspace=True)
    gt = gt.rename(columns={'ActionGeo_FullName': 'city', 'Lat': 'lat', 'Long': 'lon'})
    pos = set()
    for _, r in gt.iterrows():
        name, lat, lon = r['city'], r['lat'], r['lon']
        gn = _norm(name)
        std = n2s.get(gn)
        if std is None and gn:
            for k, v in n2s.items():
                if k and (gn in k or k in gn):
                    std = v
                    break
        if std is None and pd.notna(lat) and pd.notna(lon):
            near = sub[sub.ActionGeo_Lat.between(lat - 0.7, lat + 0.7)
                       & sub.ActionGeo_Long.between(lon - 0.7, lon + 0.7)].copy()
            if not near.empty:
                near['dist'] = haversine(lat, lon, near.ActionGeo_Lat.values, near.ActionGeo_Long.values)
                near = near[near.dist <= 50]
            if not near.empty:
                tot = near.groupby('standard_name')['NumMentions'].sum().sort_values(ascending=False)
                if tot.iloc[0] / tot.sum() >= 0.6:
                    std = tot.index[0]
        if std:
            pos.add(std)
    return pos


TIER_MENT_MIN, TIER_ACTIVE_MIN = 10, 10

raw = pd.read_parquet(MAIN_PATH)
raw['date'] = raw['SQLDATE'].astype(str).str[:8]
raw = raw[raw.EventCode.astype(int).isin(CONFIRMED_CODES)]
raw = raw[raw.ActionGeo_CountryCode.isin(MONITORED_COUNTRIES) & (raw.ActionGeo_Type == 4)]
raw = raw.dropna(subset=['ActionGeo_FullName', 'ActionGeo_Lat', 'ActionGeo_Long'])
raw = raw[~raw['ActionGeo_FullName'].isin(CITY_BLACKLIST)]
raw = apply_name_map(raw)
raw = raw[~raw['standard_name'].isin(CITY_BLACKLIST)]
raw['standard_name'] = raw['standard_name'].str.split('|').str[0]

agg = raw.groupby(['date', 'standard_name']).agg(mentions=('NumMentions', 'sum')).reset_index()
agg['date_dt'] = pd.to_datetime(agg['date'])
mnt_lookup = {(r.standard_name, r.date): int(r.mentions) for r in agg.itertuples(index=False)}
ad_lookup = {}
for city, sub in agg.groupby('standard_name'):
    dts = sorted(sub['date_dt'].tolist())
    for i, dt in enumerate(dts):
        lo = dt - pd.Timedelta(days=30)
        ad_lookup[(city, dt.strftime('%Y%m%d'))] = sum(1 for d in dts[:i] if d >= lo)

def tier_of(city, date_str):
    m = mnt_lookup.get((city, date_str), 0)
    a = ad_lookup.get((city, date_str), 0)
    return 'A' if (m >= TIER_MENT_MIN and a >= TIER_ACTIVE_MIN) else 'C'

print(f"{'date':10s} {'GT_A':>4s} {'GT_C':>4s} {'sent':>4s} {'sat':>4s} "
      f"{'TP_A':>4s} {'TP_C':>4s} {'P_LLM':>6s} {'R_A':>5s} {'R_C':>5s} {'R_ALL':>6s}")
print("-" * 90)
totals = {'gt_a': 0, 'gt_c': 0, 'sent': 0, 'sat': 0, 'tp_a': 0, 'tp_c': 0}
for d in DATES:
    pos = gt_for_date(d, raw)
    try:
        dash = json.load(open(PROJECT_ROOT / "dashboard" / f"daily_{d}.json"))
        rec  = json.load(open(PROJECT_ROOT / "records"   / f"daily_{d}.json"))
    except FileNotFoundError:
        print(f"{d:10s}  (dashboard/records JSON 없음)")
        continue
    sat_cities = {t['city'] for t in dash['targets']}
    sent_cities = sat_cities | {c['city'] for c in rec['rejected_candidates']}

    gt_a = {c for c in pos if tier_of(c, d) == 'A'}
    gt_c = pos - gt_a
    tp_a = sat_cities & gt_a
    tp_c = sat_cities & gt_c

    p_llm = len(sat_cities) / len(sent_cities) if sent_cities else 0.0
    r_a = len(tp_a) / len(gt_a) if gt_a else float('nan')
    r_c = len(tp_c) / len(gt_c) if gt_c else float('nan')
    r_all = (len(tp_a) + len(tp_c)) / len(pos) if pos else 0.0

    totals['gt_a'] += len(gt_a); totals['gt_c'] += len(gt_c)
    totals['sent'] += len(sent_cities); totals['sat'] += len(sat_cities)
    totals['tp_a'] += len(tp_a); totals['tp_c'] += len(tp_c)

    def fmt(x):
        return f"{x:>5.3f}" if x == x else "  -  "
    print(f"{d:10s} {len(gt_a):>4d} {len(gt_c):>4d} {len(sent_cities):>4d} {len(sat_cities):>4d} "
          f"{len(tp_a):>4d} {len(tp_c):>4d} {p_llm:>6.3f} {fmt(r_a)} {fmt(r_c)} {r_all:>6.3f}")

P_LLM = totals['sat'] / totals['sent'] if totals['sent'] else 0.0
R_A   = totals['tp_a'] / totals['gt_a'] if totals['gt_a'] else 0.0
R_C   = totals['tp_c'] / totals['gt_c'] if totals['gt_c'] else 0.0
R_ALL = (totals['tp_a'] + totals['tp_c']) / (totals['gt_a'] + totals['gt_c']) if (totals['gt_a']+totals['gt_c']) else 0.0
print("-" * 90)
print(f"{'TOTAL':10s} {totals['gt_a']:>4d} {totals['gt_c']:>4d} {totals['sent']:>4d} {totals['sat']:>4d} "
      f"{totals['tp_a']:>4d} {totals['tp_c']:>4d} {P_LLM:>6.3f} "
      f"{R_A:>5.3f} {R_C:>5.3f} {R_ALL:>6.3f}")
