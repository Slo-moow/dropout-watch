import sys
from pathlib import Path
import joblib, pandas as pd, streamlit as st
ROOT = Path(__file__).resolve().parents[1]          # relative to this file: no absolute paths
sys.path.insert(0, str(ROOT / "src"))
from validation import LIMITS, CATS, REQUIRED, validate_csv

st.set_page_config(page_title="Dropout Watch", page_icon="🎓")
st.title("🎓 Dropout Watch – Early Identification of At-Risk Students")

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "best_model.joblib")   # full pipeline: preprocessing + classifier
model = load_model()
MODEL_NAME = {"LogisticRegression": "Logistic Regression", "DecisionTreeClassifier": "Decision Tree",
              "RandomForestClassifier": "Random Forest"}.get(type(model.named_steps["clf"]).__name__, "Classifier")
st.caption(f"**Model in use: {MODEL_NAME}** (selected on cross-validated dropout recall). "
           "It estimates the probability that a student drops out and flags students at or above 50%.")

with st.expander("📘 Feature guide and how to read the result", expanded=False):
    rows = [(k.replace("_", " "), "number", f"{lo:g} to {hi:g}", hint) for k, (lo, hi, _d, hint) in LIMITS.items()]
    rows += [(k.replace("_", " "), "choice", ", ".join(o), "category") for k, o in CATS.items()]
    st.table(pd.DataFrame(rows, columns=["Feature", "Type", "Allowed values", "Unit / meaning"]).set_index("Feature"))
    st.markdown("**Output:** *Predicted dropout probability* is the model's estimated chance (0–100%) that the student drops out. "
                "*At risk* means the probability is 50% or higher. Fewer than half of flagged students actually drop out, "
                "so treat a flag as a prompt for a supportive conversation, not a conclusion. Empty CSV cells are filled "
                "with typical values learned from the training data.")

def predict(df):
    p = model.predict_proba(df[REQUIRED])[:, 1]
    return p, (p >= 0.5).astype(int)

tab1, tab2 = st.tabs(["Single student", "CSV upload"])
with tab1:
    with st.form("f"):
        vals = {}
        c1, c2 = st.columns(2)
        for i, (k, (lo, hi, d, hint)) in enumerate(LIMITS.items()):
            with (c1 if i % 2 == 0 else c2):
                vals[k] = st.number_input(f"{k.replace('_', ' ')} ({hint})", min_value=lo, max_value=hi, value=d)
        for i, (k, opts) in enumerate(CATS.items()):
            with (c1 if i % 2 == 0 else c2):
                vals[k] = st.selectbox(k.replace("_", " "), opts)
        go = st.form_submit_button("Predict")
    if go:
        p, y = predict(pd.DataFrame([vals]))
        st.metric("Predicted dropout probability", f"{p[0]:.1%}")
        (st.error if y[0] else st.success)("AT RISK of dropping out" if y[0] else "Not flagged as at risk")
        st.caption("Decision threshold 0.5. This is a screening aid for advisors, not a final judgement about any student.")

with tab2:
    st.write("Upload a CSV with these columns: " + ", ".join(REQUIRED) + ". Empty cells are allowed and are filled automatically.")
    template = pd.DataFrame([{k: v[2] for k, v in LIMITS.items()} | {k: o[0] for k, o in CATS.items()}])
    st.download_button("Download CSV template", template.to_csv(index=False), "dropout_watch_template.csv")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        try:
            raw = pd.read_csv(up)
        except pd.errors.EmptyDataError:
            raw = None; st.error("The file is empty. Please upload a CSV with a header row and at least one student.")
        except Exception:
            raw = None; st.error("Could not read this file as a CSV. Please check that it is comma-separated text and try again.")
        if raw is not None:
            errors, clean = validate_csv(raw)
            if errors:
                st.error("Please fix the following and upload again:")
                for e in errors:
                    st.write("• " + e)
            else:
                n_empty = int(clean[REQUIRED].isna().sum().sum())
                if n_empty:
                    st.info(f"{n_empty} empty cell(s) were filled with training-set typical values.")
                p, y = predict(clean)
                out = raw.copy(); out["Dropout_Probability"] = p.round(3); out["At_Risk"] = y
                st.dataframe(out)
                st.download_button("Download results", out.to_csv(index=False), "predictions.csv")
