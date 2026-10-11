import sys
from pathlib import Path
import joblib, pandas as pd, streamlit as st
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pipeline  # noqa: F401  (needed so joblib can unpickle the pipeline)

st.set_page_config(page_title="Dropout Watch", page_icon="🎓")
st.title("🎓 Dropout Watch – Early Identification of At-Risk Students")

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "best_model.joblib")
model = load_model()
st.caption(f"Model in use: **{type(model.named_steps['clf']).__name__}** (selected on cross-validated dropout recall; see paper).")

LIMITS = {  # (min, max, default, unit/meaning)
    "Age": (16.0, 40.0, 21.0, "years"),
    "Family_Income": (0.0, 500000.0, 38000.0, "monthly family income (currency units of the dataset)"),
    "Study_Hours_per_Day": (0.0, 16.0, 4.0, "hours per day"),
    "Attendance_Rate": (0.0, 100.0, 82.0, "% of classes attended"),
    "Assignment_Delay_Days": (0, 30, 2, "average days assignments are late"),
    "Travel_Time_Minutes": (0.0, 240.0, 30.0, "minutes one way"),
    "Stress_Index": (1.0, 10.0, 5.5, "1 = low, 10 = high"),
    "GPA": (0.0, 4.0, 2.5, "0.0 – 4.0 scale"),
}
CATS = {"Gender": ["Female", "Male"], "Internet_Access": ["Yes", "No"], "Part_Time_Job": ["No", "Yes"],
        "Scholarship": ["No", "Yes"], "Semester": ["Year 1", "Year 2", "Year 3", "Year 4"],
        "Department": ["Arts", "Business", "CS", "Engineering", "Science"],
        "Parental_Education": ["High School", "Bachelor", "Master", "PhD"]}

def predict(df):
    p = model.predict_proba(df)[:, 1]
    return p, (p >= 0.5).astype(int)

tab1, tab2 = st.tabs(["Single student", "CSV upload"])
with tab1:
    with st.form("f"):
        vals = {}
        c1, c2 = st.columns(2)
        for i, (k, (lo, hi, d, hint)) in enumerate(LIMITS.items()):
            with (c1 if i % 2 == 0 else c2):
                vals[k] = st.number_input(f"{k.replace('_',' ')} ({hint})", min_value=lo, max_value=hi, value=d)
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
    st.write("Upload a CSV with columns: " + ", ".join(list(LIMITS) + list(CATS)))
    up = st.file_uploader("CSV file", type="csv")
    if up:
        try:
            d = pd.read_csv(up)
            missing = [c for c in list(LIMITS) + list(CATS) if c not in d.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                bad = [c for c, (lo, hi, *_ ) in LIMITS.items() if ((d[c] < lo) | (d[c] > hi)).any()]
                badc = [c for c, o in CATS.items() if (~d[c].isin(o + [None]) & d[c].notna()).any()]
                if bad or badc:
                    st.error(f"Out-of-range or unknown values in: {bad + badc}")
                else:
                    p, y = predict(d[list(LIMITS) + list(CATS)])
                    d["Dropout_Probability"], d["At_Risk"] = p.round(3), y
                    st.dataframe(d); st.download_button("Download results", d.to_csv(index=False), "predictions.csv")
        except Exception as e:  # noqa
            st.error(f"Could not read the file: {e}")
