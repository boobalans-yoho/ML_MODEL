from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def find_dataset() -> Path:
    candidates = [
        DATA_DIR / "modified-spotify-dataset (1).csv",
        DATA_DIR / "modified-spotify-dataset.csv",
        DATA_DIR / "loan_data.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    raise FileNotFoundError(f"No CSV dataset could be found in {DATA_DIR}.")


def build_preprocessor(X_train: pd.DataFrame):
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = []
    if len(numeric_features) > 0:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if len(categorical_features) > 0:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


FILE_PATH = find_dataset()
df = pd.read_csv(FILE_PATH)


def main() -> None:
    print("=" * 70)
    print("TASK 3 - MODELS")
    print("=" * 70)
    print("\nDataset Shape:", df.shape)

    TARGET = "is_dance_hit"
    if TARGET not in df.columns:
        raise ValueError(f"{TARGET} not found in dataset.")

    y = df[TARGET].astype(int)

    X = df.drop(columns=[TARGET]).copy()

    genre_columns = [column for column in X.columns if column.startswith("track_genre_")]
    X = X.drop(columns=genre_columns, errors="ignore")
    X = X.drop(columns=["track_genre", "track_genre_le"], errors="ignore")
    print("\nRemoved genre columns:", len(genre_columns))

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    X = X[numeric_columns]
    print("Numeric features:", len(numeric_columns))

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=42,
        stratify=y_temp,
    )

    print("\n" + "=" * 70)
    print("DATA SPLIT")
    print("=" * 70)
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    preprocessor = build_preprocessor(X_train)

    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION")
    print("=" * 70)
    logistic_model = Pipeline(
        [
            ("preprocessing", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    logistic_model.fit(X_train, y_train)
    logistic_prediction = logistic_model.predict(X_val)
    logistic_accuracy = accuracy_score(y_val, logistic_prediction)
    print("Validation Accuracy:", round(logistic_accuracy, 4))

    print("\n" + "=" * 70)
    print("DECISION TREE")
    print("=" * 70)
    tree_model = Pipeline(
        [
            ("preprocessing", preprocessor),
            ("model", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ]
    )
    tree_model.fit(X_train, y_train)
    tree_prediction = tree_model.predict(X_val)
    tree_accuracy = accuracy_score(y_val, tree_prediction)
    print("Validation Accuracy:", round(tree_accuracy, 4))

    print("\n" + "=" * 70)
    print("RANDOM FOREST")
    print("=" * 70)
    forest_model = Pipeline(
        [
            ("preprocessing", preprocessor),
            ("model", RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)),
        ]
    )
    forest_model.fit(X_train, y_train)
    forest_prediction = forest_model.predict(X_val)
    forest_accuracy = accuracy_score(y_val, forest_prediction)
    print("Validation Accuracy:", round(forest_accuracy, 4))

    print("\n" + "=" * 70)
    print("XGBOOST")
    print("=" * 70)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)

    xgb_model = XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
    )

    xgb_model.fit(
        X_train_transformed,
        y_train,
        eval_set=[(X_val_transformed, y_val)],
        verbose=False,
    )
    xgb_prediction = xgb_model.predict(X_val_transformed)
    xgb_accuracy = accuracy_score(y_val, xgb_prediction)
    print("Validation Accuracy:", round(xgb_accuracy, 4))
    print("Best XGBoost Iteration:", xgb_model.get_booster().best_iteration)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    results = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"],
            "Validation Score": [logistic_accuracy, tree_accuracy, forest_accuracy, xgb_accuracy],
        }
    )
    print(results.to_string(index=False))
    results.to_csv(PLOTS_DIR / "task3_model_results.csv", index=False)

    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION - PREDICT_PROBA")
    print("=" * 70)

    five_rows = X_val.iloc[:5]
    five_probabilities = logistic_model.predict_proba(five_rows)
    five_predictions = logistic_model.predict(five_rows)
    print("\nProbability for 5 rows:")
    print(five_probabilities)
    print("\nPredicted class:")
    print(five_predictions)

    print("\n" + "=" * 70)
    print("TOP 5 RANDOM FOREST FEATURES")
    print("=" * 70)
    forest_preprocessor = forest_model.named_steps["preprocessing"]
    forest_classifier = forest_model.named_steps["model"]
    forest_feature_names = forest_preprocessor.get_feature_names_out()
    forest_importances = forest_classifier.feature_importances_
    forest_feature_table = pd.DataFrame({"Feature": forest_feature_names, "Importance": forest_importances})
    forest_feature_table = forest_feature_table.sort_values("Importance", ascending=False)
    print(forest_feature_table.head(5).to_string(index=False))
    forest_feature_table.to_csv(PLOTS_DIR / "random_forest_feature_importance.csv", index=False)

    print("\n" + "=" * 70)
    print("TOP 5 XGBOOST FEATURES")
    print("=" * 70)
    xgb_feature_names = preprocessor.get_feature_names_out()
    xgb_importances = xgb_model.feature_importances_
    xgb_feature_table = pd.DataFrame({"Feature": xgb_feature_names, "Importance": xgb_importances})
    xgb_feature_table = xgb_feature_table.sort_values("Importance", ascending=False)
    print(xgb_feature_table.head(5).to_string(index=False))
    xgb_feature_table.to_csv(PLOTS_DIR / "xgboost_feature_importance.csv", index=False)

    print("\n" + "=" * 70)
    print("REGRESSION")
    print("=" * 70)
    REGRESSION_TARGET = "popularity"
    if REGRESSION_TARGET not in df.columns:
        raise ValueError("Popularity column not found.")

    regression_df = df.loc[df[REGRESSION_TARGET].notna(), :]
    X_reg = regression_df.drop(columns=[REGRESSION_TARGET, TARGET])
    X_reg = X_reg.drop(columns=genre_columns, errors="ignore")
    X_reg = X_reg.drop(columns=["track_genre", "track_genre_le"], errors="ignore")
    X_reg = X_reg.select_dtypes(include=["number"])
    y_reg = regression_df[REGRESSION_TARGET]

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42,
    )

    regression_preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    linear_model = Pipeline(
        [
            ("preprocessing", regression_preprocessor),
            ("model", LinearRegression()),
        ]
    )
    linear_model.fit(X_reg_train, y_reg_train)
    linear_prediction = linear_model.predict(X_reg_test)
    linear_mae = mean_absolute_error(y_reg_test, linear_prediction)
    linear_rmse = np.sqrt(mean_squared_error(y_reg_test, linear_prediction))
    linear_r2 = r2_score(y_reg_test, linear_prediction)

    ridge_model = Pipeline(
        [
            ("preprocessing", regression_preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    ridge_model.fit(X_reg_train, y_reg_train)
    ridge_prediction = ridge_model.predict(X_reg_test)
    ridge_mae = mean_absolute_error(y_reg_test, ridge_prediction)
    ridge_rmse = np.sqrt(mean_squared_error(y_reg_test, ridge_prediction))
    ridge_r2 = r2_score(y_reg_test, ridge_prediction)

    lasso_model = Pipeline(
        [
            ("preprocessing", regression_preprocessor),
            ("model", Lasso(alpha=0.1, max_iter=10000)),
        ]
    )
    lasso_model.fit(X_reg_train, y_reg_train)
    lasso_prediction = lasso_model.predict(X_reg_test)
    lasso_mae = mean_absolute_error(y_reg_test, lasso_prediction)
    lasso_rmse = np.sqrt(mean_squared_error(y_reg_test, lasso_prediction))
    lasso_r2 = r2_score(y_reg_test, lasso_prediction)

    regression_results = pd.DataFrame(
        {
            "Model": ["Linear Regression", "Ridge", "Lasso"],
            "MAE": [linear_mae, ridge_mae, lasso_mae],
            "RMSE": [linear_rmse, ridge_rmse, lasso_rmse],
            "R2": [linear_r2, ridge_r2, lasso_r2],
        }
    )
    print(regression_results.to_string(index=False))
    regression_results.to_csv(PLOTS_DIR / "task3_regression_results.csv", index=False)

    print("\n" + "=" * 70)
    print("REGRESSION COEFFICIENTS")
    print("=" * 70)
    coefficient_table = pd.DataFrame(
        {
            "Feature": X_reg.columns,
            "Linear": linear_model.named_steps["model"].coef_,
            "Ridge": ridge_model.named_steps["model"].coef_,
            "Lasso": lasso_model.named_steps["model"].coef_,
        }
    )
    print(coefficient_table.to_string(index=False))
    print("\nLasso zero coefficients:", np.sum(lasso_model.named_steps["model"].coef_ == 0))

    print("\n" + "=" * 70)
    print("K-MEANS CLUSTERING")
    print("=" * 70)
    cluster_features = [column for column in ["danceability", "energy", "valence"] if column in df.columns]
    if len(cluster_features) < 2:
        raise ValueError("Not enough clustering columns.")
    print("Clustering Features:", cluster_features)
    cluster_data = df[cluster_features].copy().fillna(df[cluster_features].median())
    cluster_scaler = StandardScaler()
    cluster_scaled = cluster_scaler.fit_transform(cluster_data)
    cluster_sample = cluster_data.sample(n=min(len(cluster_data), 3000), random_state=42)
    cluster_sample_scaled = cluster_scaler.fit_transform(cluster_sample)

    inertias = []
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(cluster_scaled)
        inertias.append(kmeans.inertia_)

    print("\nElbow Inertia:")
    for k, inertia in zip(range(2, 9), inertias):
        print(f"K={k}: {inertia:.2f}")

    plt.figure(figsize=(9, 6))
    plt.plot(range(2, 9), inertias, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("K-Means Elbow Curve")
    plt.grid(True)
    plt.savefig(PLOTS_DIR / "task3_elbow_curve.png", dpi=300, bbox_inches="tight")

    silhouette_scores = []
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(cluster_sample_scaled)
        score = silhouette_score(cluster_sample_scaled, labels)
        silhouette_scores.append(score)
        print(f"K={k}: Silhouette Score={score:.4f}")

    plt.figure(figsize=(9, 6))
    plt.plot(range(2, 9), silhouette_scores, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("K-Means Silhouette Scores")
    plt.grid(True)
    plt.savefig(PLOTS_DIR / "task3_silhouette_scores.png", dpi=300, bbox_inches="tight")

    best_k_index = int(np.argmax(silhouette_scores))
    best_k = 2 + best_k_index
    best_silhouette = silhouette_scores[best_k_index]
    print("\n" + "=" * 70)
    print("BEST K")
    print("=" * 70)
    print("Best K:", best_k)
    print("Best Silhouette Score:", round(best_silhouette, 4))

    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = final_kmeans.fit_predict(cluster_scaled)
    cluster_summary = df.groupby("cluster")[cluster_features].mean()
    print("\n" + "=" * 70)
    print("CLUSTER SEGMENTS")
    print("=" * 70)
    print(cluster_summary)
    cluster_summary.to_csv(PLOTS_DIR / "task3_cluster_summary.csv")

    print("\n" + "=" * 70)
    print("TASK 3 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
