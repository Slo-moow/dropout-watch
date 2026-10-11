"""EDA on the TRAINING portion's source data; figures saved to documentation/figures."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, pandas as pd, numpy as np
from pipeline import ROOT
df = pd.read_csv(ROOT / "data/original/student_dropout_dataset_v3.csv")
out = ROOT / "documentation/figures"; out.mkdir(exist_ok=True, parents=True)
sns.set_theme(style="whitegrid", font_scale=0.9)
def save(n): plt.tight_layout(); plt.savefig(out / n, dpi=200); plt.close()

# Fig 1: (a) target balance, (b) missing values
fig, ax = plt.subplots(1, 2, figsize=(3.5, 2.1))
sns.countplot(x="Dropout", data=df, palette=["#4C78A8", "#E45756"], hue="Dropout", legend=False, ax=ax[0])
for p in ax[0].patches: ax[0].annotate(f"{p.get_height()/len(df):.1%}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom", fontsize=7)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["Retained", "Dropout"]); ax[0].set_xlabel("(a) Target balance"); ax[0].set_ylim(0, 9000)
miss = df.isna().sum(); miss = miss[miss > 0].sort_values()
ax[1].barh([m.replace("_", " ") for m in miss.index], miss.values, color="#72B7B2"); ax[1].set_xlabel("(b) Missing values"); ax[1].tick_params(axis="y", labelsize=6)
save("fig1_target_balance.png")

# Fig 3: key distributions by class (stacked for column width)
fig, ax = plt.subplots(3, 1, figsize=(3.5, 4.0))
for a_, c in zip(ax, ["GPA", "Attendance_Rate", "Stress_Index"]):
    sns.kdeplot(data=df, x=c, hue="Dropout", common_norm=False, fill=True, ax=a_, palette=["#4C78A8", "#E45756"], legend=False); a_.set_ylabel("Density")
ax[0].legend(["Dropout", "Retained"], fontsize=6); save("fig3_key_distributions.png")

# Fig 4 correlation heatmap
num = df.drop(columns=["Student_ID"]).select_dtypes("number"); num.columns = [c.replace("_", " ")[:14] for c in num.columns]
plt.figure(figsize=(3.5, 3.2)); sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 5}, cbar=False)
plt.xticks(fontsize=5, rotation=60, ha="right"); plt.yticks(fontsize=5); save("fig4_correlation.png")

# Fig 5 income distribution (floor/outliers)
fig, ax = plt.subplots(2, 1, figsize=(3.5, 2.8))
sns.histplot(df.Family_Income.dropna(), bins=50, ax=ax[0], color="#F58518"); ax[0].set_xlabel("Family income (histogram)")
sns.boxplot(x=df.Family_Income, ax=ax[1], color="#F58518"); ax[1].set_xlabel("Family income (boxplot)")
save("fig5_income_outliers.png")

# Fig 6 dropout rate by categorical variables
cats = ["Part_Time_Job", "Internet_Access", "Semester", "Parental_Education"]
fig, ax = plt.subplots(2, 2, figsize=(3.5, 3.0))
for a_, c in zip(ax.ravel(), cats):
    r = df.assign(**{c: df[c].fillna("Missing")}).groupby(c).Dropout.mean().sort_values(); r.plot.barh(ax=a_, color="#9D755D"); a_.set_title(c.replace("_", " "), fontsize=7)
    a_.set_xlabel("Dropout rate", fontsize=6); a_.set_ylabel(""); a_.tick_params(labelsize=6); a_.axvline(df.Dropout.mean(), color="k", ls="--", lw=0.7)
save("fig6_categorical_rates.png")

# numeric facts for the paper
facts = {"rows": len(df), "dropout_rate": df.Dropout.mean(), "income_at_floor_25000": int((df.Family_Income == 25000).sum()),
         "age_min": df.Age.min(), "age_max": df.Age.max(), "gpa_cgpa_corr": df.GPA.corr(df.CGPA), "semgpa_cgpa_corr": df.Semester_GPA.corr(df.CGPA)}
pd.Series(facts).to_csv(ROOT / "documentation/eda_facts.csv"); print(pd.Series(facts))
print(df.groupby("Semester").Dropout.mean().round(3).to_dict(), df.groupby("Scholarship").Dropout.mean().round(3).to_dict(),
      df.groupby("Part_Time_Job").Dropout.mean().round(3).to_dict(), df.groupby("Department").Dropout.mean().round(3).to_dict())
