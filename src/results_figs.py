"""Results figures for the paper (uses saved model + stored CV table; no re-selection)."""
import json, joblib, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from pipeline import ROOT
sns.set_theme(style="whitegrid", font_scale=0.9)
out = ROOT / "documentation/figures"
cv = pd.read_csv(ROOT / "documentation/cv_results.csv"); test = json.load(open(ROOT / "documentation/test_results.json"))
# Fig 7: CV comparison (mean +- std)
mets = ["recall", "precision", "f1_macro", "roc_auc"]; lab = ["Recall", "Precision", "Macro F1", "ROC AUC"]
fig, ax = plt.subplots(figsize=(3.5, 2.6)); w = 0.26
for i, (_, r) in enumerate(cv.iterrows()):
    ax.bar(np.arange(4) + (i - 1) * w, [r[m + "_mean"] for m in mets], w, yerr=[r[m + "_std"] for m in mets], label=r["Model"], capsize=2)
ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=7); ax.set_ylim(0.3, 1.1); ax.legend(fontsize=6, loc="upper center", ncol=3); ax.set_ylabel("5-fold CV score")
plt.tight_layout(); plt.savefig(out / "fig7_cv_comparison.png", dpi=200); plt.close()
# Fig 8: test confusion matrix
cm = np.array(test["confusion_matrix"]); plt.figure(figsize=(3.0, 2.6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Retained", "Dropout"], yticklabels=["Retained", "Dropout"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.savefig(out / "fig8_confusion_matrix.png", dpi=200); plt.close()
# Fig 9: LR coefficients
m = joblib.load(ROOT / "models/best_model.joblib"); names = m.named_steps["prep"].get_feature_names_out(); coef = m.named_steps["clf"].coef_[0]
s = pd.Series(coef, index=[n.split("__")[1] for n in names]).sort_values(); s.to_csv(ROOT / "documentation/lr_coefficients.csv")
top = pd.concat([s.head(5), s.tail(5)]); plt.figure(figsize=(3.5, 2.8)); top.plot.barh(color=["#4C78A8" if v < 0 else "#E45756" for v in top])
plt.xlabel("Coefficient (log-odds)", fontsize=7); plt.tight_layout(); plt.savefig(out / "fig9_lr_coefficients.png", dpi=200); plt.close()
print(s.round(3).to_string()); print(m.named_steps["clf"].get_params())
