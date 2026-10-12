# Dropout Watch – Early Identification of At-Risk Students

An ML-powered student dropout prediction system that identifies students at risk of dropping out and supports early intervention through data-driven insights.

This project compares Logistic Regression, Decision Tree, and Random Forest models for binary classification of student dropout outcomes.

## Setup

```bash
pip install -r requirements.txt
python src/eda.py
python src/train.py
streamlit run app/streamlit_app.py
```

Run these commands from the project root directory.

## Model Design

- **Data split:** 80/20 stratified train-test split using `random_state=42`.
- **Cross-validation:** 5-fold stratified cross-validation on the training set, using the same folds for all models.
- **Preprocessing:** Median/mode imputation, scaling, and one-hot encoding within the scikit-learn pipeline to prevent data leakage.
- **Model selection:** Cross-validated dropout-class recall, with hyperparameters tuned using macro F1.
- **Class balancing:** `class_weight="balanced"` for all three models.
- **Testing:** The selected model is evaluated once on the held-out test set.
- **Excluded features:** `Student_ID`, `Semester_GPA`, and `CGPA`, due to identifier concerns and high correlation with GPA, respectively.

## Project Structure

- `app/` – Streamlit application
- `data/original/` – Original dataset
- `data/processed/` – Processed training and testing data
- `src/` – Exploratory data analysis, model training, and evaluation scripts
- `models/` – Saved trained model
- `documentation/` – Results, metrics, and visualizations
- `paper/` – Research paper and supporting files

## Deployment

The application can be deployed using Streamlit Community Cloud.

1. Push the project to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app and select this repository, the `main` branch, and `app/streamlit_app.py` as the main file.
4. Follow the deployment instructions and add the deployed URL here when available.

**Deployed URL:** [deployed](https://dropout-watch-nms3rgi5d9tnxddzwrou6v.streamlit.app/).
