"""Shared preprocessing and model definitions (used by training AND the app)."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
TARGET = "Dropout"
# Student_ID = identifier; Semester_GPA/CGPA dropped (r>=0.96 with GPA, near-duplicates)
DROP_COLS = ["Student_ID", "Semester_GPA", "CGPA"]
NUM = ["Age", "Family_Income", "Study_Hours_per_Day", "Attendance_Rate",
       "Assignment_Delay_Days", "Travel_Time_Minutes", "Stress_Index", "GPA"]
CAT = ["Gender", "Internet_Access", "Part_Time_Job", "Scholarship",
       "Semester", "Department", "Parental_Education"]

def load_data(path=None):
    df = pd.read_csv(path or ROOT / "data/original/student_dropout_dataset_v3.csv")
    return df.drop(columns=DROP_COLS)

def preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])

def models_and_grids():
    cw = "balanced"
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight=cw, random_state=RANDOM_STATE),
                                {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l1", "l2"], "clf__solver": ["liblinear"]}),
        "Decision Tree": (DecisionTreeClassifier(class_weight=cw, random_state=RANDOM_STATE),
                          {"clf__max_depth": [3, 5, 7, 10], "clf__min_samples_leaf": [2, 10, 30]}),
        "Random Forest": (RandomForestClassifier(n_estimators=200, class_weight=cw, random_state=RANDOM_STATE, n_jobs=1),
                          {"clf__max_depth": [5, 10, None], "clf__min_samples_leaf": [5, 20]}),
    }

def full_pipeline(clf):
    return Pipeline([("prep", preprocessor()), ("clf", clf)])
