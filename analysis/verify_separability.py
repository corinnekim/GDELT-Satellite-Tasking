"""우선순위 2: 사건 vs 평시의 z-score 분리도.

논리:
  N(0,1) 정규성은 위반됐지만, ranking 능력은 별개.
  ROC AUC, PR AUC, Youden's J 최적 threshold로 입증.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

OUT = Path('/Users/corinnekim/gdelt/analysis/results')
arr = np.load(OUT / 'innovation_arrays.npz')
z_peace, z_event = arr['z_peace'], arr['z_event']

y_true = np.concatenate([np.zeros(len(z_peace)), np.ones(len(z_event))])
y_score = np.concatenate([z_peace, z_event])

# ROC
fpr, tpr, thr_roc = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# PR
prec, rec, thr_pr = precision_recall_curve(y_true, y_score)
pr_auc = average_precision_score(y_true, y_score)

# Youden J
J = tpr - fpr
j_best = J.argmax()
z_youden = thr_roc[j_best]
tpr_y = tpr[j_best]; fpr_y = fpr[j_best]

# Operating points: z=0.5, 2, 5 (현행 risk thresholds)
print(f"\n분리도 결과")
print("─"*64)
print(f"{'metric':28s} {'value':>10s}")
print("─"*64)
print(f"{'ROC AUC':28s} {roc_auc:>10.4f}")
print(f"{'PR AUC':28s} {pr_auc:>10.4f}")
print(f"{'baseline pos rate':28s} {y_true.mean():>10.4f}")
print(f"{'PR AUC / baseline':28s} {pr_auc/y_true.mean():>10.2f}x")
print(f"{'Youden J optimal z':28s} {z_youden:>10.3f}")
print(f"  → TPR':28s {tpr_y:>10.4f}")
print(f"  → FPR':28s {fpr_y:>10.4f}")
print("─"*64)
print(f"{'operating points':28s}")
print(f"{'thr':>6s} {'TPR(recall)':>12s} {'FPR':>10s} {'precision':>12s}")
for thr in [0.5, 2.0, 5.0, 10.0, 20.0]:
    tp = (z_event >= thr).sum()
    fp = (z_peace >= thr).sum()
    fn = (z_event < thr).sum()
    tn = (z_peace < thr).sum()
    tpr_t = tp / max(tp + fn, 1)
    fpr_t = fp / max(fp + tn, 1)
    prec_t = tp / max(tp + fp, 1)
    print(f"{thr:>6.1f} {tpr_t:>12.4f} {fpr_t:>10.4f} {prec_t:>12.4f}")
print("─"*64)

# plots
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
ax.plot(fpr, tpr, color='#5fdfa8', lw=2, label=f'ROC AUC = {roc_auc:.3f}')
ax.plot([0,1], [0,1], 'k--', lw=0.8, alpha=0.5)
ax.scatter([fpr_y], [tpr_y], color='#e25a4a', s=80, zorder=5, label=f"Youden's J  z*={z_youden:.2f}")
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title(f'(a) ROC — peace vs event innovation z')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(rec, prec, color='#c4702c', lw=2, label=f'PR AUC = {pr_auc:.3f}')
ax.axhline(y_true.mean(), color='gray', ls='--', lw=0.8, label=f'baseline = {y_true.mean():.4f}')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title(f'(b) PR — lift {pr_auc/y_true.mean():.1f}x over baseline')
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
fig_path = OUT / 'separability.png'
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"\nsaved: {fig_path}")
