import pandas as pd

# ============================================================
# 1) LOAD DATA
# ============================================================
def load_data(path: str) -> pd.DataFrame:
    print(f"[LOAD] Reading data from: {path}")
    df = pd.read_csv(path)
    print(f"[LOAD] Loaded shape: {df.shape}")
    return df


# ============================================================
# 2) CLEAN DATA
# ============================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print(f"[CLEAN] Starting shape: {df.shape}")

    # Drop identifier column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        print("[CLEAN] Dropped customerID")

    # Fix TotalCharges
    if "TotalCharges" in df.columns:
        before_missing = df["TotalCharges"].isna().sum()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        after_missing = df["TotalCharges"].isna().sum()
        print(
            f"[CLEAN] TotalCharges converted to numeric "
            f"(missing before: {before_missing}, after: {after_missing})"
        )

    # Drop missing values
    before_rows = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped_rows = before_rows - len(df)
    print(f"[CLEAN] Dropped {dropped_rows} rows with missing values")
    print(f"[CLEAN] Shape after cleaning: {df.shape}")

    return df


# ============================================================
# 3) TARGET PROCESSING
# ============================================================
def process_target(df: pd.DataFrame, target_col: str):
    df = df.copy()

    if target_col in df.columns:
        print(f"[TARGET] Encoding target column: {target_col}")
        print("[TARGET] Before encoding:")
        print(df[target_col].value_counts())

        df[target_col] = df[target_col].map({"No": 0, "Yes": 1})

        print("[TARGET] After encoding:")
        print(df[target_col].value_counts(dropna=False))

    return df


# ============================================================
# 4) FEATURE ENGINEERING
# ============================================================
def feature_engineering(df):
    df = df.copy()
    print("[FEATURE] Starting feature engineering")

    # 1. Avg. monthly charges
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["avg_monthly_charge"] = df["TotalCharges"] / (df["tenure"] + 1)
        print("[FEATURE] Created avg_monthly_charge")

    # 2. Binning tenure into groups for easy splitting in decision trees
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 60, float("inf")],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5+yr"]
        )
        print("[FEATURE] Created tenure_group")

    # 3. Flag if contract is month-to-month (this is a strong predictor of churn)
    if "Contract" in df.columns:
        df["is_monthly_contract"] = (df["Contract"] == "Month-to-month").astype(int)
        print("[FEATURE] Created is_monthly_contract")

    print(f"[FEATURE] Shape after feature engineering: {df.shape}")
    return df


# ============================================================
# 5) ENCODING
# ============================================================
def encode_features(df: pd.DataFrame, target_col: str):
    df = df.copy()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"[ENCODE] Feature shape before encoding: {X.shape}")
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"[ENCODE] Categorical columns: {cat_cols}")

    X = pd.get_dummies(X, drop_first=True)

    # cols_to_drop = [col for col in X.columns if "StreamingTV" in col or "StreamingMovies" in col]
    # X = X.drop(columns=cols_to_drop)

    print(f"[ENCODE] Feature shape after encoding: {X.shape}")
    return X, y


# ============================================================
# 6) FULL PIPELINE
# ============================================================
def prepare_data(path: str, target_col: str = "Churn"):
    print("[PIPELINE] Starting data preparation")

    df = load_data(path)
    df = clean_data(df)
    df = process_target(df, target_col)
    df = feature_engineering(df)

    X, y = encode_features(df, target_col)

    print("[PIPELINE] Data preparation complete")
    print(f"[PIPELINE] Final X shape: {X.shape}")
    print(f"[PIPELINE] Final y shape: {y.shape}")

    return X, y, df

prepare_data("../data/telco.csv", target_col="Churn")