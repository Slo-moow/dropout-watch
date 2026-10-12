"""Input limits and CSV validation shared by the Streamlit app and the tests."""
import pandas as pd

# feature: (min, max, default, unit/meaning)
LIMITS = {
    "Age": (16.0, 40.0, 21.0, "years"),
    "Family_Income": (0.0, 500000.0, 38000.0, "family income, same currency units as the dataset"),
    "Study_Hours_per_Day": (0.0, 16.0, 4.0, "hours per day"),
    "Attendance_Rate": (0.0, 100.0, 82.0, "% of classes attended"),
    "Assignment_Delay_Days": (0.0, 30.0, 2.0, "average days assignments are late"),
    "Travel_Time_Minutes": (0.0, 240.0, 30.0, "minutes, one way"),
    "Stress_Index": (1.0, 10.0, 5.5, "1 = low stress, 10 = high stress"),
    "GPA": (0.0, 4.0, 2.5, "0.0 to 4.0 scale"),
}
CATS = {
    "Gender": ["Female", "Male"], "Internet_Access": ["Yes", "No"], "Part_Time_Job": ["No", "Yes"],
    "Scholarship": ["No", "Yes"], "Semester": ["Year 1", "Year 2", "Year 3", "Year 4"],
    "Department": ["Arts", "Business", "CS", "Engineering", "Science"],
    "Parental_Education": ["High School", "Bachelor", "Master", "PhD"],
}
REQUIRED = list(LIMITS) + list(CATS)

def _rows(mask, n=5):
    r = [str(i + 2) for i in mask[mask].index[:n]]          # +2: header row + 1-based, as in Excel
    return ", ".join(r) + (" ..." if mask.sum() > n else "")

def validate_csv(d: pd.DataFrame):
    """Return (errors, cleaned_df). cleaned_df has numeric columns coerced; empty cells stay NaN
    (the saved pipeline imputes them with training-set statistics)."""
    if d.empty:
        return ["The file has no data rows."], None
    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        return [f"Missing required column(s): {', '.join(missing)}. Download the template to see the expected format."], None
    d = d.copy(); errors = []
    for c, (lo, hi, *_rest) in LIMITS.items():
        num = pd.to_numeric(d[c], errors="coerce")
        text = d[c].notna() & num.isna()
        if text.any():
            errors.append(f"Column '{c}' must be a number; found text in row(s): {_rows(text)}.")
        out = num.notna() & ((num < lo) | (num > hi))
        if out.any():
            errors.append(f"Column '{c}' must be between {lo:g} and {hi:g}; out-of-range value(s) in row(s): {_rows(out)}.")
        d[c] = num
    for c, opts in CATS.items():
        bad = d[c].notna() & ~d[c].isin(opts)
        if bad.any():
            errors.append(f"Column '{c}' allows only {', '.join(opts)}; unknown value(s) in row(s): {_rows(bad)}.")
    return errors, (None if errors else d)
