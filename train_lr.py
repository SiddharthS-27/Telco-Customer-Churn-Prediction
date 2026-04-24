import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

from preprocess import prepare_data


# 1) Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/telco.csv")
MLFLOW_DB = os.path.join(BASE_DIR, "../mlflow.db")


# 2) Mlflow setup
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment("telco_churn_experiment_logistic_regression")


# 3) Data loading and preprocessing
TARGET_COL = "Churn"
X, y, raw_df = prepare_data(DATA_PATH, target_col=TARGET_COL)

print("\n[DATA]")
print(f"Raw dataframe shape: {raw_df.shape}")
print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Class distribution:\n{y.value_counts()}")


# 4) Train-test split
test_size = 0.2
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y,
)

print("\n[SPLIT]")
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")


# =========================
# MODEL CONFIG
# =========================

# ----- RANDOM FOREST (existing) -----
# MODEL_NAME = "random_forest"
# MODEL_PARAMS = {
#     "n_estimators": 100,
#     "max_depth": 14,
#     "min_samples_leaf": 5,
#     "random_state": 42,
#     "class_weight": "balanced"
# }

# ----- LOGISTIC REGRESSION (NEW) -----
MODEL_NAME = "logistic_regression"

MODEL_PARAMS = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "solver": "liblinear",
    "random_state": 42,
}


def get_model(name, params):
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)

    elif name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params)

    elif name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**params)

    else:
        raise ValueError(f"Unknown model: {name}")


def get_score_vector(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


# =========================
# SCALING (ONLY FOR LOGISTIC REGRESSION)
# =========================
if MODEL_NAME == "logistic_regression":
    from sklearn.preprocessing import StandardScaler

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])


# 6) Baseline metrics
majority_class = y_train.value_counts().idxmax()
baseline_pred = [majority_class] * len(y_test)

baseline_accuracy = accuracy_score(y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred, zero_division=0)
baseline_precision = precision_score(y_test, baseline_pred, zero_division=0)
baseline_recall = recall_score(y_test, baseline_pred, zero_division=0)

print("\n[BASELINE]")
print(f"Majority class: {majority_class}")
print(f"Baseline accuracy: {baseline_accuracy:.4f}")
print(f"Baseline f1: {baseline_f1:.4f}")
print(f"Baseline precision: {baseline_precision:.4f}")
print(f"Baseline recall: {baseline_recall:.4f}")


# 7) Training, evaluation, and mlflow logging
run_name = f"{MODEL_NAME}_without_feature_engineering"
mlflow.set_tag("model_type", "logistic_regression")
mlflow.set_tag("preprocessing", "without_feature_engineering")

if mlflow.active_run():
    mlflow.end_run()

with mlflow.start_run(run_name=run_name):

    mlflow.log_param("dataset_name", "telco_churn")
    mlflow.log_param("dataset_rows", len(raw_df))
    mlflow.log_param("dataset_features", X.shape[1])
    mlflow.log_param("target_column", TARGET_COL)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("class_0_count", int((y == 0).sum()))
    mlflow.log_param("class_1_count", int((y == 1).sum()))

    mlflow.log_metric("baseline_accuracy", baseline_accuracy)
    mlflow.log_metric("baseline_f1", baseline_f1)
    mlflow.log_metric("baseline_precision", baseline_precision)
    mlflow.log_metric("baseline_recall", baseline_recall)

    mlflow.log_param("model_name", MODEL_NAME)
    for param, value in MODEL_PARAMS.items():
        mlflow.log_param(param, value)

    model = get_model(MODEL_NAME, MODEL_PARAMS)

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_scores = get_score_vector(model, X_train)
    test_scores = get_score_vector(model, X_test)

    # TRAIN METRICS
    train_accuracy = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred, zero_division=0)
    train_precision = precision_score(y_train, train_pred, zero_division=0)
    train_recall = recall_score(y_train, train_pred, zero_division=0)

    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("train_f1", train_f1)
    mlflow.log_metric("train_precision", train_precision)
    mlflow.log_metric("train_recall", train_recall)

    if train_scores is not None:
        try:
            train_roc_auc = roc_auc_score(y_train, train_scores)
            mlflow.log_metric("train_roc_auc", train_roc_auc)
        except Exception as e:
            print(f"[WARN] Could not compute train ROC-AUC: {e}")

    # TEST METRICS
    acc = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    precision = precision_score(y_test, test_pred, zero_division=0)
    recall = recall_score(y_test, test_pred, zero_division=0)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    if test_scores is not None:
        try:
            roc_auc = roc_auc_score(y_test, test_scores)
            mlflow.log_metric("roc_auc", roc_auc)
        except Exception as e:
            print(f"[WARN] Could not compute test ROC-AUC: {e}")

    cm_filename = f"cm_{MODEL_NAME}.png"
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.tight_layout()
    plt.savefig(cm_filename)
    mlflow.log_artifact(cm_filename)
    plt.close()

    mlflow.sklearn.log_model(model, "model")

    print(f"\n[{MODEL_NAME}] TEST RESULTS")
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test F1 Score : {f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall   : {recall:.4f}")
    if test_scores is not None:
        try:
            print(f"Test ROC-AUC  : {roc_auc:.4f}")
        except NameError:
            pass