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
mlflow.set_experiment("telco_churn_experiment")


# 3) Data loading and preprocessing
TARGET_COL = "Churn"
X, y, raw_df = prepare_data(DATA_PATH, target_col=TARGET_COL)

print("\n[DATA]")
print(f"Raw dataframe shape: {raw_df.shape}")
print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}") # shape of target column
print(f"Class distribution:\n{y.value_counts()}")


# 4) Train-test split

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier

print("\n[CROSS-VALIDATION]")

cv_model = HistGradientBoostingClassifier(
    learning_rate=0.03,
    max_depth=6,
    max_iter=100,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_f1 = cross_val_score(cv_model, X, y, cv=cv, scoring="f1")

print("F1 scores per fold:", cv_f1)
print("Mean CV F1:", cv_f1.mean())
print("Std CV F1:", cv_f1.std())

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


# 5) Model and training setup
# MODEL_NAME = "random_forest"

# MODEL_PARAMS = {
#     "n_estimators": 100, # no. of trees in the forest
#     "max_depth": 14, # max depth of each tree
#     "min_samples_leaf": 5, # min samples required to be at a leaf node (prevents overfitting by ensuring leaves have enough samples)
#     "random_state": 42, # for reproducibility (this decides randomness in the bootstrap samples and feature selection at each split)
#     "class_weight": "balanced", # penalises minority class errors more to handle class imbalance
# }

MODEL_NAME = "hist_gb"

MODEL_PARAMS = {
    "learning_rate": 0.03,
    "max_depth": 6,
    "max_iter": 100,
    "random_state": 42
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
    
    elif name == "hist_gb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(**params)
    
    # elif name == "ensemble":
    #     from sklearn.ensemble import VotingClassifier, RandomForestClassifier, HistGradientBoostingClassifier

    #     rf = RandomForestClassifier(
    #         n_estimators=100,
    #         max_depth=14,
    #         min_samples_leaf=5,
    #         random_state=42,
    #         class_weight="balanced"
    #     )

    #     hgb = HistGradientBoostingClassifier(
    #         learning_rate=0.03,
    #         max_depth=6,
    #         max_iter=200,
    #         random_state=42
    #     )

    #     return VotingClassifier(
    #         estimators=[
    #             ("rf", rf),
    #             ("hgb", hgb)
    #         ],
    #         voting="soft"
    #     )

    else:
        raise ValueError(f"Unknown model: {name}")


def get_score_vector(model, X):
    """
    Returns a continuous score for ROC-AUC.
    Uses predict_proba if available, otherwise decision_function.
    """
    if hasattr(model, "predict_proba"): # for each sample, gives probability of belonging to each class. We take the probability of the positive class ([:, 1]). Gives continuous scores between 0 and 1, which is ideal for ROC-AUC.
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):# outputs real values that represent the distance of the samples to the decision boundary. Negative = class 0, Positive = class 1, Magnitude = confidence
        return model.decision_function(X)
    return None


# 6) Baseline metrics (model has to beat this to be useful, else we might as well predict the majority class every time)
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
# run_name = (
#     f"{MODEL_NAME}"
#     f"_d{MODEL_PARAMS['max_depth']}"
#     f"_leaf{MODEL_PARAMS.get('min_samples_leaf', 1)}"
#     f"_n{MODEL_PARAMS.get('n_estimators', 'na')}"
#     f"{'_cw_bal' if MODEL_PARAMS.get('class_weight') == 'balanced' else ''}"
# )

run_name = (
    f"{MODEL_NAME}"
    f"_lr{MODEL_PARAMS.get('learning_rate', 'na')}"
    f"_d{MODEL_PARAMS.get('max_depth', 'na')}"
    f"_iter{MODEL_PARAMS.get('max_iter', 'na')}"
)

with mlflow.start_run(run_name=run_name):
    mlflow.set_tag("stage", "final")
    mlflow.set_tag("model", "hist_gb")
    mlflow.set_tag("status", "selected")
    # LOGGING EXPERIMENT SETUP
    mlflow.log_param("dataset_name", "telco_churn")
    mlflow.log_param("dataset_rows", len(raw_df))
    mlflow.log_param("dataset_features", X.shape[1])
    mlflow.log_param("target_column", TARGET_COL)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("class_0_count", int((y == 0).sum()))
    mlflow.log_param("class_1_count", int((y == 1).sum()))

    # LOGGING BASELINE METRICS
    mlflow.log_metric("baseline_accuracy", baseline_accuracy)
    mlflow.log_metric("baseline_f1", baseline_f1)
    mlflow.log_metric("baseline_precision", baseline_precision)
    mlflow.log_metric("baseline_recall", baseline_recall)

    mlflow.log_param("model_name", MODEL_NAME)
    for param, value in MODEL_PARAMS.items():
        mlflow.log_param(param, value)

    model = get_model(MODEL_NAME, MODEL_PARAMS)

    from sklearn.utils.class_weight import compute_sample_weight

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )
    
    # model.fit(X_train, y_train)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # # =========================
    # # FEATURE IMPORTANCE
    # # =========================

    # feature_importances = model.feature_importances_ # gives score based on how much each feature reduces impurity (Gini) across all trees at different splits
    # feature_names = X_train.columns

    # fi_df = pd.DataFrame({
    #     "feature": feature_names,
    #     "importance": feature_importances
    # }).sort_values(by="importance", ascending=False)

    # fi_df = fi_df.reset_index(drop=True)

    # print("\n[FEATURE IMPORTANCE - TOP 20]")
    # print(fi_df.head(20))

    # print("\n[FEATURE IMPORTANCE - BOTTOM 20]")
    # print(fi_df.tail(20))
    from sklearn.inspection import permutation_importance

    print("\n[FEATURE IMPORTANCE - PERMUTATION]")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="f1"
    )

    importances = pd.Series(result.importances_mean, index=X.columns)
    importances = importances.sort_values(ascending=False)

    print("\nTop 15 Features:")
    print(importances.head(15))

    plt.figure(figsize=(10, 6))
    importances.head(15).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Top 15 Feature Importances (Permutation)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    fi_filename = "feature_importance.png"
    plt.savefig(fi_filename)
    mlflow.log_artifact(fi_filename)
    plt.close()

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_scores = get_score_vector(model, X_train) # getting continuous scores for train set (either proba or decision_function for use in ROC-AUC)
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

    # LOGGING CONFUSION MATRIX
    cm_filename = f"cm_{MODEL_NAME}.png"
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.tight_layout()
    plt.savefig(cm_filename)
    mlflow.log_artifact(cm_filename)
    plt.close()

    # LOGGING MODEL
    mlflow.sklearn.log_model(model, "model")

    print(f"\n[{MODEL_NAME}] TRAIN RESULTS")
    print(f"Train Accuracy : {train_accuracy:.4f}")
    print(f"Train F1 Score : {train_f1:.4f}")
    print(f"Train Precision: {train_precision:.4f}")
    print(f"Train Recall   : {train_recall:.4f}")
    if train_scores is not None:
        try:
            print(f"Train ROC-AUC  : {train_roc_auc:.4f}")
        except NameError:
            pass

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