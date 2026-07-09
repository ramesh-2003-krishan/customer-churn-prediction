import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

# 1. Load dataset
print("Loading dataset...")
df = pd.read_csv("data/telco_customer_churn.csv")

# 2. Fix Total Charges datatype and check missing values
# Total Charges is stored as a string with blank entries. Convert to numeric and fill with 0 (new customers).
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0)

# Check missing values
print("\nMissing values after cleanup:")
print(df.isnull().sum())

# Define columns to drop
leakage_cols = ["Churn Value", "Churn Score", "CLTV"]
unnecessary_cols = [
    "CustomerID", "Count", "Country", "State", "City", 
    "Zip Code", "Lat Long", "Latitude", "Longitude", "Churn Reason"
]

# --- Training Baseline Model with Data Leakage for Demonstration ---
print("\n" + "="*50)
print("  TRAINING BASELINE MODEL (WITH DATA LEAKAGE)")
print("="*50)

df_leaked = df.drop(unnecessary_cols, axis=1)
df_leaked["Churn Label"] = df_leaked["Churn Label"].map({"Yes": 1, "No": 0})
df_leaked = pd.get_dummies(df_leaked, drop_first=True)

X_leak = df_leaked.drop("Churn Label", axis=1)
y_leak = df_leaked["Churn Label"]

# Stratified split for leaked model
X_train_leak, X_test_leak, y_train_leak, y_test_leak = train_test_split(
    X_leak, y_leak, test_size=0.2, random_state=42, stratify=y_leak
)

# Fit simple RandomForestClassifier
model_leak = RandomForestClassifier(random_state=42)
model_leak.fit(X_train_leak, y_train_leak)
preds_leak = model_leak.predict(X_test_leak)
probs_leak = model_leak.predict_proba(X_test_leak)[:, 1]

print("\n[Baseline Leaked Model] Test Set Metrics:")
print(f"Accuracy:  {accuracy_score(y_test_leak, preds_leak):.4f}")
print(f"Precision: {precision_score(y_test_leak, preds_leak):.4f}")
print(f"Recall:    {recall_score(y_test_leak, preds_leak):.4f}")
print(f"F1-score:  {f1_score(y_test_leak, preds_leak):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test_leak, probs_leak):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_leak, preds_leak))
print("\nClassification Report (Leaked Model):")
print(classification_report(y_test_leak, preds_leak))


# --- Training Corrected Model (Without Data Leakage) ---
print("\n" + "="*50)
print("  TRAINING CORRECTED MODEL (WITHOUT DATA LEAKAGE)")
print("="*50)

# Drop both unnecessary and leaked columns
df_clean = df.drop(unnecessary_cols + leakage_cols, axis=1)
df_clean["Churn Label"] = df_clean["Churn Label"].map({"Yes": 1, "No": 0})
df_clean = pd.get_dummies(df_clean, drop_first=True)

X_clean = df_clean.drop("Churn Label", axis=1)
y_clean = df_clean["Churn Label"]

# Save feature column names for dashboard tracking
joblib.dump(X_clean.columns.tolist(), "model_columns.pkl")
print(f"Saved {len(X_clean.columns)} model columns to model_columns.pkl")

# Stratified split for corrected model
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
)

# Hyperparameter Tuning using Stratified K-Fold CV & GridSearchCV
# Using RandomForestClassifier with class_weight='balanced' to handle class imbalance
rf = RandomForestClassifier(class_weight="balanced", random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 4]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

print("Starting grid search hyperparameter tuning...")
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

# Compute 5-fold cross-validation scores using StratifiedKFold
cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
print(f"\n5-Fold CV F1-Scores on Training Set: {cv_scores}")
print(f"Mean CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

preds = best_model.predict(X_test)
probs = best_model.predict_proba(X_test)[:, 1]

# In-depth Evaluation Metrics
print("\n[Corrected Model] Test Set Metrics:")
print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
print(f"Precision: {precision_score(y_test, preds):.4f}")
print(f"Recall:    {recall_score(y_test, preds):.4f}")
print(f"F1-score:  {f1_score(y_test, preds):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, probs):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))
print("\nClassification Report (Corrected Model):")
print(classification_report(y_test, preds))

# Save corrected model
joblib.dump(best_model, "churn_model.pkl")
print("Corrected model saved successfully to churn_model.pkl")

# WHY ACCURACY IS MISLEADING (EXPLANATION IN COMMENT):
"""
Why accuracy is misleading for class-imbalanced datasets:
The target variable (Churn) is imbalanced: ~73% is "No Churn" and ~27% is "Churn".
If we used a naive classifier that predicts "No Churn" for every customer, it would achieve 
73% accuracy. Yet, the model would fail completely (0% recall) in identifying any churning customers, 
which is the key business goal. 
Relying on accuracy obscures poor performance on the minority class. F1-score (balance of precision 
and recall) and ROC-AUC are much better indicators of model performance on imbalanced datasets.
To address this, we set 'class_weight="balanced"' in the RandomForest Classifier and tuned for F1-score.

Alternative Approach (SMOTE):
Instead of class_weight="balanced" (cost-sensitive learning), another solution is synthetic minority 
oversampling (SMOTE) from the `imbalanced-learn` library. 
To use SMOTE, we install `imbalanced-learn` and run:
--------------------------------------------------
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
# Then fit the classifier on (X_train_res, y_train_res)
--------------------------------------------------
This makes the training distribution 50/50 by creating synthetic instances of churn. 
However, class cost weights are simpler and do not require generating synthetic data points.
"""

# Feature Importance Logging
importances = best_model.feature_importances_
feature_names = X_clean.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10)

print("\n" + "="*50)
print("  TOP 10 FEATURE IMPORTANCES (CORRECTED MODEL)")
print("="*50)
for idx, row in feature_imp_df.iterrows():
    print(f"{row['Feature']:<30} : {row['Importance']:.4f}")

# Plot top 10 feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nFeature importance plot saved to 'feature_importance.png'")
