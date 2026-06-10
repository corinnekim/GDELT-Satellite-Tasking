"""kalman_fig1_baseline.png 재생성 — 가독성 개선판.

좌: Tehran/Beirut/Natanz raw conflict index (같은 스케일)
우: 동일 도시들의 표준화 이노베이션 (35σ 클리핑)

개선점:
- 폰트 키우고
- annotation에 흰 배경 box 추가 (라인 위에 있어도 글자가 가려지지 않음)
- annotation 위치를 데이터와 안 겹치게 조정
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pipeline.config import (MAIN_PATH, MONITORED_COUNTRIES, CITY_BLACKLIST,
                              CONFIRMED_CODES, KALMAN_Q, KALMAN_R, KALMAN_MIN_VAR,
                              KALMAN_P0_RATIO)
from pipeline.preprocess import apply_name_map

OUT = Path('/Users/corinnekim/gdelt/report/images/kalman_fig1_baseline.png')


def kalman_run(signal, Q=KALMAN_Q, R=KALMAN_R):
    n = len(signal)
    init_var = max(np.var(signal[:min(30, n)]), KALMAN_MIN_VAR)
    x = np.zeros(n); P = np.zeros(n); innov = np.zeros(n); znorm = np.zeros(n)
    x[0] = signal[0]
    P[0] = R * KALMAN_P0_RATIO
    for t in range(1, n):
        x_pred = x[t-1]
        P_pred = P[t-1] + Q
        innov[t] = signal[t] - x_pred
        S = P_pred + R
        znorm[t] = innov[t] / max(np.sqrt(S), 1e-10)
        K = P_pred / S
        x[t] = x_pred + K * innov[t]
        P[t] = (1 - K) * P_pred
    return x, innov, znorm


# ── Load + build conflict index ──────────────────────────────────────────
raw = pd.read_parquet(MAIN_PATH)
raw['date'] = raw['SQLDATE'].astype(str).str[:8]
raw = raw[raw.EventCode.astype(int).isin(CONFIRMED_CODES)]
raw = raw[raw.ActionGeo_CountryCode.isin(MONITORED_COUNTRIES) & (raw.ActionGeo_Type == 4)]
raw = raw.dropna(subset=['ActionGeo_FullName', 'ActionGeo_Lat', 'ActionGeo_Long', 'NumSources'])
raw = raw[~raw['ActionGeo_FullName'].isin(CITY_BLACKLIST)]
raw = apply_name_map(raw)
raw = raw[~raw['standard_name'].isin(CITY_BLACKLIST)]
raw['standard_name'] = raw['standard_name'].str.split('|').str[0]

raw['z_e'] = np.log1p(raw['NumSources'].astype(float))
agg = raw.groupby(['date','standard_name'])['z_e'].sum().reset_index()
agg['date_dt'] = pd.to_datetime(agg['date'])

# ── Build per-city series with full date range, fill 0 ───────────────────
date_range = pd.date_range(agg['date_dt'].min(), agg['date_dt'].max(), freq='D')
cities = [('Tehran', 'Tehran'), ('Beirut', 'Beirut'), ('Natanz', 'Naţanz')]
colors = {'Tehran': '#d62728', 'Beirut': '#1f77b4', 'Natanz': '#2ca02c'}
series = {}
for label, sn in cities:
    sub = agg[agg['standard_name']==sn].set_index('date_dt')['z_e']
    sub = sub.reindex(date_range, fill_value=0.0)
    series[label] = sub
cities = [c for c, _ in cities]

# ── Run kalman ───────────────────────────────────────────────────────────
results = {}
for c, s in series.items():
    sig = s.values.astype(np.float64)
    _, _, znorm = kalman_run(sig)
    results[c] = {'signal': sig, 'znorm': znorm, 'dates': s.index}

# ── Plot ─────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
                     'legend.fontsize': 11, 'figure.titlesize': 16})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
fig.suptitle('Kalman Filter — Why City-Specific Baseline Adaptation is Essential',
             fontweight='bold', y=1.0)

# Left panel: raw conflict index
for c in cities:
    avg = results[c]['signal'].mean()
    ax1.plot(results[c]['dates'], results[c]['signal'],
             color=colors[c], linewidth=1.4,
             label=f"{c} (avg={avg:.0f})")
ax1.set_title('Raw Conflict Index — same scale')
ax1.set_ylabel('Conflict index   ($\\Sigma$ log(1 + NumSources))')
ax1.legend(loc='upper left', framealpha=0.95)
ax1.grid(True, alpha=0.3)

# Annotations on left panel — with white bbox so visible over lines
peak_tehran = results['Tehran']['signal'].max()
peak_natanz = results['Natanz']['signal'].max()
bbox_kw = dict(boxstyle='round,pad=0.4', facecolor='white',
               edgecolor='gray', alpha=0.92)
ax1.annotate(f'Tehran peak: {peak_tehran:,.0f}\nany global threshold\nset here always fires',
             xy=(results['Tehran']['dates'][np.argmax(results['Tehran']['signal'])], peak_tehran),
             xytext=(0.04, 0.78), textcoords='axes fraction',
             color='#d62728', fontsize=10, fontweight='bold',
             bbox=bbox_kw,
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2))
ax1.annotate(f'Natanz spikes to {peak_natanz:.0f}\nbut crushed by Tehran scale',
             xy=(results['Natanz']['dates'][np.argmax(results['Natanz']['signal'])], peak_natanz),
             xytext=(0.42, 0.40), textcoords='axes fraction',
             color='#2ca02c', fontsize=10, fontweight='bold',
             bbox=bbox_kw,
             arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.2))

# Right panel: standardised innovation, clipped at 35
CLIP = 35
ax2.set_title('Standardised Innovation $\\tilde{\\nu}_t = \\nu_t / \\sqrt{S_t}$\n'
              'All cities on the same scale — clipped at 35$\\sigma$')
ax2.set_ylabel('$\\sigma$ above city-specific baseline')

for c in cities:
    z = results[c]['znorm']
    z_clip = np.clip(z, -CLIP, CLIP)
    peak = z.max()
    over = (z > CLIP).sum()
    label = f"{c} (peak={peak:.0f}σ"
    if over > 0:
        label += f", {over}d off-chart"
    label += ")"
    ax2.plot(results[c]['dates'], z_clip,
             color=colors[c], linewidth=1.4, label=label)

# Reference threshold lines
for thr, lbl, ls in [(5, 'RED z=5', '--'), (2, 'ORANGE z=2', ':')]:
    ax2.axhline(thr, color='gray', linestyle=ls, linewidth=1, alpha=0.7)
    ax2.text(results['Tehran']['dates'][-1], thr+0.5, lbl,
             color='gray', fontsize=9, ha='right', va='bottom')

ax2.set_ylim(-5, CLIP + 2)
ax2.legend(loc='upper left', framealpha=0.95)
ax2.grid(True, alpha=0.3)

# Annotation on right: Natanz detection
natanz_peak_idx = np.argmax(results['Natanz']['znorm'])
natanz_peak_z = results['Natanz']['znorm'][natanz_peak_idx]
ax2.annotate(f'Natanz {natanz_peak_z:.0f}σ\nclearly detected\ndespite small raw value',
             xy=(results['Natanz']['dates'][natanz_peak_idx], min(natanz_peak_z, CLIP)),
             xytext=(0.55, 0.55), textcoords='axes fraction',
             color='#2ca02c', fontsize=10, fontweight='bold',
             bbox=bbox_kw,
             arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.2))

# X-axis date formatting
import matplotlib.dates as mdates
for ax in (ax1, ax2):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=[1, 15]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches='tight')
print(f"Saved: {OUT}")

# Print summary stats for caption
print("\nSummary:")
for c in cities:
    sig = results[c]['signal']
    z = results[c]['znorm']
    print(f"  {c}: avg raw={sig.mean():.1f}, peak raw={sig.max():.0f}, peak z={z.max():.1f}")
