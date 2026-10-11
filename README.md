# Dropout Watch – Early Identification of At-Risk Students
Binary classification (Dropout 0/1) comparing Logistic Regression, Decision Tree and Random Forest.

## Setup
    pip install -r requirements.txt
    cd src && python eda.py && python train.py     # figures, CV table, test results, saved model
    streamlit run app/streamlit_app.py             # run from the project root

## Design
- Split 80/20 stratified, random_state=42; 5-fold stratified CV on the training set only; same folds for all models.
- Preprocessing (median/mode imputation, scaling, one-hot) lives inside the sklearn Pipeline, fit on training folds only.
- Selection metric: cross-validated dropout-class recall (hyperparameters tuned on macro F1 to avoid degenerate settings). class_weight="balanced" for all three models.
- Test set is evaluated once, on the selected model only (documentation/test_results.json).
- Dropped columns: Student_ID (identifier), Semester_GPA and CGPA (r >= 0.96 with GPA).

## Folders
app/ · data/original · data/processed · src/ · models/ · documentation/ · paper/
Deployed URL: [INSERT STREAMLIT URL]

## Deploy (Streamlit Community Cloud)
1. Push this whole folder to a public GitHub repo (keep models/best_model.joblib, src/, app/ and requirements.txt).
2. share.streamlit.io -> New app -> pick the repo, branch, and main file `app/streamlit_app.py`. Use Python 3.12.
3. Paste the resulting URL here and in the paper: [INSERT STREAMLIT URL]
