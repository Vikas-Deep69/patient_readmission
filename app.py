# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Load Dataset
df = pd.read_csv("hospital_readmissions.csv")

# Basic Info
df.info()
df.describe()

# Missing Values
print(df.isnull().sum())

# Readmission Distribution
sns.countplot(x='readmitted', data=df)
plt.title('Readmitted Class Distribution')
plt.xlabel('Readmitted (0=No, 1=Yes)')
plt.ylabel('Count')
plt.show()

# Histograms for Numeric Features
numeric_features = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications']
df[numeric_features].hist(figsize=(10,8), bins=15)
plt.suptitle('Distribution of Numeric Features')
plt.show()

# Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder

label_encoders_file = "label_encoders.pkl"
label_encoders_2_file = "label_encoders_2.pkl"

# Load or Fit Label Encoders
if os.path.exists(label_encoders_file) and os.path.exists(label_encoders_2_file):
    print("Loading existing label encoders...")
    label_encoders = joblib.load(label_encoders_file)
    label_encoders_2 = joblib.load(label_encoders_2_file)
else:
    print("Fitting new label encoders...")
    categorical_cols = ['age', 'glucose_test', 'A1Ctest', 'change', 'diabetes_med', 'readmitted']
    label_encoders = {col: LabelEncoder() for col in categorical_cols}

    for col, le in label_encoders.items():
        df[col] = le.fit_transform(df[col].astype(str))

    categorical_cols_2 = ['medical_specialty', 'diag_1', 'diag_2', 'diag_3']
    label_encoders_2 = {col: LabelEncoder() for col in categorical_cols_2}

    for col2, le in label_encoders_2.items():
        df[col2] = le.fit_transform(df[col2].astype(str))

    joblib.dump(label_encoders, label_encoders_file)
    joblib.dump(label_encoders_2, label_encoders_2_file)
    print("Label encoders saved.")

# Feature Engineering (Interaction Terms)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
interaction_terms = poly.fit_transform(df[['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications']])
interaction_df = pd.DataFrame(interaction_terms, columns=poly.get_feature_names_out(
    ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications']))
df = pd.concat([df, interaction_df], axis=1)

# Splitting the Data
from sklearn.model_selection import train_test_split

X = df.drop('readmitted', axis=1)
y = df['readmitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Remove duplicate columns
X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]


feature_columns_file = "feature_columns.pkl"
joblib.dump(X_train.columns.tolist(), feature_columns_file)
print(f"Feature columns saved as {feature_columns_file}")

# Define Model Filenames
rf_model_file = "rf_tuned_model.pkl"
xgb_model_file = "xgb_model.pkl"
lgbm_model_file = "lgbm_model.pkl"

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

if os.path.exists(rf_model_file):
    print(f"Loading existing Random Forest model from {rf_model_file}...")
    rf_model = joblib.load(rf_model_file)
else:
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        bootstrap=True,
        max_depth=10,
        min_samples_leaf=4,
        min_samples_split=5,
        n_estimators=200,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, rf_model_file)
    print(f"Random Forest model saved as {rf_model_file}")

# XGBoost Classifier
from xgboost import XGBClassifier

if os.path.exists(xgb_model_file):
    print(f"Loading existing XGBoost model from {xgb_model_file}...")
    xgb_model = joblib.load(xgb_model_file)
else:
    print("Training XGBoost model...")
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, xgb_model_file)
    print(f"XGBoost model saved as {xgb_model_file}")

# LightGBM Classifier
from lightgbm import LGBMClassifier

if os.path.exists(lgbm_model_file):
    print(f"Loading existing LightGBM model from {lgbm_model_file}...")
    lgbm_model = joblib.load(lgbm_model_file)
else:
    print("Training LightGBM model...")
    lgbm_model = LGBMClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    lgbm_model.fit(X_train, y_train)
    joblib.dump(lgbm_model, lgbm_model_file)
    print(f"LightGBM model saved as {lgbm_model_file}")

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

y_pred_lgbm = lgbm_model.predict(X_test)
y_proba_lgbm = lgbm_model.predict_proba(X_test)[:, 1]

# Evaluation
from sklearn.metrics import classification_report, roc_auc_score

print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest ROC-AUC Score:", roc_auc_score(y_test, y_proba_rf))

print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("XGBoost ROC-AUC Score:", roc_auc_score(y_test, y_proba_xgb))

print("LightGBM Classification Report:\n", classification_report(y_test, y_pred_lgbm))
print("LightGBM ROC-AUC Score:", roc_auc_score(y_test, y_proba_lgbm))

# Compare Models
results = {
    "Model": ["Random Forest", "XGBoost", "LightGBM"],
    "ROC-AUC Score": [roc_auc_score(y_test, y_proba_rf),
                      roc_auc_score(y_test, y_proba_xgb),
                      roc_auc_score(y_test, y_proba_lgbm)]
}

results_df = pd.DataFrame(results)
print("\nModel Performance Summary:")
print(results_df)

# Plot Model Comparison
sns.barplot(data=results_df, x="Model", y="ROC-AUC Score", palette="viridis")
plt.title("Model ROC-AUC Comparison")
plt.ylabel("ROC-AUC Score")
plt.show()
