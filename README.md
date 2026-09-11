# Predictive-Churn-Analysis
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# ==============================================================================
# Step 1: Dataset Ingestion & Feature Engineering Setup
# ==============================================================================
# Checks for local dataset (e.g. churn.csv); generates representative data if absent
csv_file = "churn.csv"

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    np.random.seed(42)
    n = 3000
    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{10000 + i}" for i in range(n)],
            "tenure_months": np.random.randint(1, 72, size=n),
            "monthly_charges": np.random.uniform(20.0, 120.0, size=n).round(2),
            "total_charges": np.random.uniform(100.0, 7000.0, size=n).round(2),
            "contract_type": np.random.choice(
                ["Month-to-month", "One year", "Two year"],
                size=n,
                p=[0.55, 0.25, 0.20],
            ),
            "internet_service": np.random.choice(
                ["DSL", "Fiber optic", "No"], size=n, p=[0.4, 0.45, 0.15]
            ),
            "payment_method": np.random.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer",
                    "Credit card",
                ],
                size=n,
            ),
            "tech_support": np.random.choice(
                ["Yes", "No", "No internet"], size=n, p=[0.3, 0.55, 0.15]
            ),
            "churn": np.random.choice([0, 1], size=n, p=[0.73, 0.27]),
        }
    )

print("Dataset Shape:", df.shape)

# Define feature columns
id_col = "customer_id"
target_col = "churn"

num_features = ["tenure_months", "monthly_charges", "total_charges"]
cat_features = [
    "contract_type",
    "internet_service",
    "payment_method",
    "tech_support",
]

X = df[num_features + cat_features].copy()
y = df[target_col].copy()

# Step 1 Pipeline: Feature Encoding (One-Hot) & Scaling (MinMax)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), num_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_features),
    ]
)

# ==============================================================================
# Step 2: 80/20 Train-Test Split (Stratified)
# ==============================================================================
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X,
    y,
    df[id_col],
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples\n")

# ==============================================================================
# Step 3: Train Logistic Regression & Random Forest Classifiers
# ==============================================================================
# 1. Logistic Regression Model
lr_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
    ]
)
lr_pipeline.fit(X_train, y_train)

# 2. Random Forest Classifier
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=7,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)
rf_pipeline.fit(X_train, y_train)

# ==============================================================================
# Step 4: Model Evaluation (Precision, Recall, F1, ROC-AUC)
# ==============================================================================
models = {
    "Logistic Regression": lr_pipeline,
    "Random Forest Classifier": rf_pipeline,
}

eval_results = []
y_probs = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_probs[name] = y_prob

    eval_results.append(
        {
            "Model": name,
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1-Score": round(f1_score(y_test, y_pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        }
    )

eval_df = pd.DataFrame(eval_results)
print("--- MODEL PERFORMANCE BENCHMARK ---")
print(eval_df.to_string(index=False))

# Select the best model based on ROC-AUC
best_model_name = eval_df.sort_values(by="ROC-AUC", ascending=False).iloc[0][
    "Model"
]
best_pipeline = models[best_model_name]
print(f"\n[✓] Optimal Selected Model: {best_model_name}")

# Visualizing ROC-AUC and Confusion Matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Plot ROC Curves
for name, y_prob in y_probs.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    score = roc_auc_score(y_test, y_prob)
    ax1.plot(fpr, tpr, label=f"{name} (AUC = {score:.3f})")

ax1.plot([0, 1], [0, 1], "k--", label="Random Chance")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC-AUC Comparison")
ax1.legend(loc="lower right")
ax1.grid(True, linestyle=":", alpha=0.6)

# Plot Confusion Matrix for Selected Best Model
cm = confusion_matrix(y_test, best_pipeline.predict(X_test))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax2,
    xticklabels=["Retained", "Churned"],
    yticklabels=["Retained", "Churned"],
)
ax2.set_xlabel("Predicted Label")
ax2.set_ylabel("Actual Label")
ax2.set_title(f"Confusion Matrix: {best_model_name}")

plt.tight_layout()
plt.savefig("model_evaluation_report.png", dpi=300)
plt.close()
print("[✓] Evaluation plot saved as 'model_evaluation_report.png'")

# ==============================================================================
# Step 5: Export Customer Churn Risk Score Predictions
# ==============================================================================
# Score the test cohort with calibrated risk probabilities
test_scores_df = pd.DataFrame(
    {
        "customer_id": id_test,
        "actual_churn": y_test,
        "churn_probability": best_pipeline.predict_proba(X_test)[:, 1].round(4),
        "predicted_label": best_pipeline.predict(X_test),
    }
)

# Categorize into Actionable Risk Tiers
test_scores_df["risk_tier"] = pd.cut(
    test_scores_df["churn_probability"],
    bins=[0.0, 0.35, 0.70, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"],
)

# Save deliverables
output_csv = "customer_churn_risk_scores.csv"
test_scores_df.sort_values(by="churn_probability", ascending=False).to_csv(
    output_csv, index=False
)
print(f"[✓] Churn risk score predictions exported to: {output_csv}")
