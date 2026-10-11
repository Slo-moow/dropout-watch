"""Split -> tune (CV on train only) -> compare -> test ONCE on selected model -> save."""
import json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
from pipeline import *

df = load_data()
X, y = df.drop(columns=[TARGET]), df[TARGET]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)   # same folds for all models
TUNE_METRIC = "f1_macro"   # tuning metric (prevents degenerate "flag everyone" settings)
PRIMARY = "recall"          # model-selection metric: dropout-class recall (false negatives are costliest)
scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1_dropout": "f1", "f1_macro": "f1_macro", "roc_auc": "roc_auc"}

rows, best = [], {}
for name, (clf, grid) in models_and_grids().items():
    gs = GridSearchCV(full_pipeline(clf), grid, scoring=TUNE_METRIC, cv=cv, n_jobs=-1).fit(Xtr, ytr)
    best[name] = gs.best_estimator_
    r = cross_validate(gs.best_estimator_, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=-1)
    row = {"Model": name, "Best params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}}
    for m in scoring:
        row[f"{m}_mean"], row[f"{m}_std"] = r[f"test_{m}"].mean(), r[f"test_{m}"].std()
    rows.append(row); print(name, gs.best_params_, round(row[f"{PRIMARY}_mean"], 4))

cvdf = pd.DataFrame(rows)
cvdf.to_csv(ROOT / "documentation/cv_results.csv", index=False)
winner = cvdf.sort_values(f"{PRIMARY}_mean", ascending=False).iloc[0]["Model"]
model = best[winner]

# ---- single test-set evaluation of the SELECTED model only ----
pred, proba = model.predict(Xte), model.predict_proba(Xte)[:, 1]
test = {"selected_model": winner, "primary_metric": PRIMARY, "tuning_metric": TUNE_METRIC,
        "accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred),
        "recall": recall_score(yte, pred), "f1_dropout": f1_score(yte, pred),
        "f1_macro": f1_score(yte, pred, average="macro"), "roc_auc": roc_auc_score(yte, proba),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
        "n_train": len(Xtr), "n_test": len(Xte), "random_state": RANDOM_STATE}
json.dump(test, open(ROOT / "documentation/test_results.json", "w"), indent=2)
print(classification_report(yte, pred, digits=3)); print(test)

joblib.dump(model, ROOT / "models/best_model.joblib")
Xtr.assign(Dropout=ytr).to_csv(ROOT / "data/processed/train_split.csv", index=False)
Xte.assign(Dropout=yte).to_csv(ROOT / "data/processed/test_split.csv", index=False)
