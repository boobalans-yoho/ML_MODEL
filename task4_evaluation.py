from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X_train.select_dtypes(include=["int64", "float64", "int32", "float32", "bool"]).columns.tolist()
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, numeric_columns)],
        remainder="drop",
    )


def evaluate_classifier(name: str, model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, zero_division=0),
        "Recall": recall_score(y_val, y_pred, zero_division=0),
        "F1": f1_score(y_val, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_val, y_prob),
    }

    print(f"\n----------------------------------------")
    print(name)
    print("----------------------------------------")
    for metric_name, value in metrics.items():
        if metric_name == "Model":
            continue
        print(f"{metric_name}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, zero_division=0))

    return metrics


def main() -> None:
    print("\n================ DATASET =================")
    file_path = find_dataset()
    df = pd.read_csv(file_path)
    print("Dataset Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("First 5 rows:\n", df.head())

    target = "is_dance_hit"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found in the dataset.")

    print("\nTarget distribution:")
    print(df[target].value_counts())

    leakage_columns = []
    if "track_genre" in df.columns:
        leakage_columns.append("track_genre")
    leakage_columns.extend([column for column in df.columns if column.startswith("track_genre_")])
    leakage_columns = list(dict.fromkeys(leakage_columns))
    print("\nColumns removed to prevent leakage:")
    print(leakage_columns)
    df_model = df.drop(columns=leakage_columns, errors="ignore")

    X = df_model.drop(columns=[target])
    y = df_model[target]
    X = X.select_dtypes(include=["int64", "float64", "int32", "float32", "bool"]).copy()
    print("\nNumber of features:", X.shape[1])

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
    print("\n================ DATA SPLIT =================")
    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    preprocessor = build_preprocessor(X_train)

    logistic_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    decision_tree_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ]
    )
    random_forest_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    xgb_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models = {
        "Logistic Regression": logistic_model,
        "Decision Tree": decision_tree_model,
        "Random Forest": random_forest_model,
        "XGBoost": xgb_model,
    }

    print("\n================ MODEL TRAINING =================")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        print(f"{name} training completed.")

    results = []
    print("\n================ CLASSIFICATION RESULTS =================")
    for name, model in models.items():
        result = evaluate_classifier(name, model, X_val, y_val)
        results.append(result)

    results_df = pd.DataFrame(results)
    print("\n================ MODEL COMPARISON =================")
    print(results_df)
    results_df.to_csv(PLOTS_DIR / "task4_classification_metrics.csv", index=False)

    print("\n================ ACCURACY EXPLANATION =================")
    print(
        "\nAccuracy can be misleading when the target classes are imbalanced. "
        "If 90% of songs are class 0, a model predicting class 0 for every row can still "
        "achieve 90% accuracy. That is why precision, recall, F1-score, and ROC-AUC are also "
        "important for evaluating classification performance."
    )

    print("\n================ ROC CURVES =================")
    roc_models = ["Logistic Regression", "Random Forest", "XGBoost"]
    plt.figure(figsize=(10, 7))
    for name in roc_models:
        model = models[name]
        probabilities = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, probabilities)
        auc_value = roc_auc_score(y_val, probabilities)
        plt.plot(fpr, tpr, label=f"{name} AUC = {auc_value:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Classification Models")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_DIR / "task4_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n================ THRESHOLD TUNING =================")
    rf_model = models["Random Forest"]
    rf_probability = rf_model.predict_proba(X_val)[:, 1]
    threshold_results = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        threshold_prediction = (rf_probability >= threshold).astype(int)
        threshold_results.append(
            {
                "Threshold": round(float(threshold), 2),
                "Precision": precision_score(y_val, threshold_prediction, zero_division=0),
                "Recall": recall_score(y_val, threshold_prediction, zero_division=0),
                "F1": f1_score(y_val, threshold_prediction, zero_division=0),
            }
        )
    threshold_df = pd.DataFrame(threshold_results)
    print(threshold_df)

    required_recall = 0.75
    valid_thresholds = threshold_df[threshold_df["Recall"] >= required_recall]
    if not valid_thresholds.empty:
        best_threshold_row = valid_thresholds.sort_values(by=["Precision", "F1"], ascending=False).iloc[0]
        best_threshold = float(best_threshold_row["Threshold"])
        print(f"\nSelected threshold: {best_threshold}")
        print(f"Precision: {best_threshold_row['Precision']:.4f}")
        print(f"Recall: {best_threshold_row['Recall']:.4f}")
        print(f"F1: {best_threshold_row['F1']:.4f}")
    else:
        best_threshold = 0.50
        print("\nNo threshold achieved 75% recall.")
        print("Using default threshold = 0.50")
    threshold_df.to_csv(PLOTS_DIR / "task4_threshold_results.csv", index=False)

    print("\n================ GRIDSEARCHCV =================")
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
            ),
        ]
    )
    rf_parameters = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
    }
    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_parameters,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    print("Best Random Forest Parameters:")
    print(grid_search.best_params_)
    print("Best Cross Validation ROC-AUC:", round(grid_search.best_score_, 4))

    print("\n================ RANDOMIZEDSEARCHCV =================")
    xgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)),
        ]
    )
    xgb_parameters = {
        "model__n_estimators": [100, 200, 300, 400],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
    }
    random_search = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=xgb_parameters,
        n_iter=15,
        cv=3,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    random_search.fit(X_train, y_train)
    print("Best XGBoost Parameters:")
    print(random_search.best_params_)
    print("Best Cross Validation ROC-AUC:", round(random_search.best_score_, 4))

    print("\n================ TUNED MODEL RESULTS =================")
    tuned_models = {
        "Tuned Random Forest": grid_search.best_estimator_,
        "Tuned XGBoost": random_search.best_estimator_,
    }
    tuned_results = []
    for name, model in tuned_models.items():
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        tuned_results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_val, y_pred),
                "Precision": precision_score(y_val, y_pred, zero_division=0),
                "Recall": recall_score(y_val, y_pred, zero_division=0),
                "F1": f1_score(y_val, y_pred, zero_division=0),
                "ROC_AUC": roc_auc_score(y_val, y_prob),
            }
        )
    tuned_results_df = pd.DataFrame(tuned_results)
    print(tuned_results_df)

    best_original = max(results, key=lambda x: x["ROC_AUC"])
    best_tuned = max(tuned_results, key=lambda x: x["ROC_AUC"])
    print("\n================ BEST MODELS =================")
    print("Best original model:", best_original["Model"])
    print("Original ROC-AUC:", round(best_original["ROC_AUC"], 4))
    print("Best tuned model:", best_tuned["Model"])
    print("Tuned ROC-AUC:", round(best_tuned["ROC_AUC"], 4))

    final_model = grid_search.best_estimator_ if best_tuned["Model"] == "Tuned Random Forest" else random_search.best_estimator_
    final_prediction = final_model.predict(X_test)
    final_probability = final_model.predict_proba(X_test)[:, 1]
    final_accuracy = accuracy_score(y_test, final_prediction)
    final_precision = precision_score(y_test, final_prediction, zero_division=0)
    final_recall = recall_score(y_test, final_prediction, zero_division=0)
    final_f1 = f1_score(y_test, final_prediction, zero_division=0)
    final_auc = roc_auc_score(y_test, final_probability)
    final_cm = confusion_matrix(y_test, final_prediction)

    print("\n================ FINAL TEST RESULTS =================")
    print("Accuracy:", round(final_accuracy, 4))
    print("Precision:", round(final_precision, 4))
    print("Recall:", round(final_recall, 4))
    print("F1 Score:", round(final_f1, 4))
    print("ROC-AUC:", round(final_auc, 4))
    print("\nConfusion Matrix:")
    print(final_cm)
    print("\nClassification Report:")
    print(classification_report(y_test, final_prediction, zero_division=0))

    print("\n================ BONUS REGRESSION =================")
    regression_target = "popularity"
    if regression_target in df.columns:
        regression_df = df.copy()
        X_reg = regression_df.drop(columns=[regression_target])
        y_reg = regression_df[regression_target]
        X_reg = X_reg.select_dtypes(include=["int64", "float64", "int32", "float32", "bool"]).copy()

        X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
            X_reg,
            y_reg,
            test_size=0.20,
            random_state=42,
        )
        regression_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
        regression_pipeline.fit(X_reg_train, y_reg_train)
        regression_prediction = regression_pipeline.predict(X_reg_test)
        mae = mean_absolute_error(y_reg_test, regression_prediction)
        rmse = np.sqrt(mean_squared_error(y_reg_test, regression_prediction))
        r2 = r2_score(y_reg_test, regression_prediction)
        print("\nLinear Regression Results")
        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("R2:", round(r2, 4))

        pd.DataFrame([
            {"Model": "Linear Regression", "MAE": mae, "RMSE": rmse, "R2": r2}
        ]).to_csv(PLOTS_DIR / "task4_regression_results.csv", index=False)
    else:
        print(f"Regression target '{regression_target}' was not found.")

    final_results = pd.DataFrame([
        {
            "Model": best_tuned["Model"],
            "Test_Accuracy": final_accuracy,
            "Test_Precision": final_precision,
            "Test_Recall": final_recall,
            "Test_F1": final_f1,
            "Test_ROC_AUC": final_auc,
        }
    ])
    final_results.to_csv(PLOTS_DIR / "final_test_results.csv", index=False)

    print("\n==============================================")
    print("TASK 4 COMPLETED SUCCESSFULLY")
    print("==============================================")
    print("\nGenerated files:")
    for name in [
        "task4_classification_metrics.csv",
        "task4_roc_curves.png",
        "task4_threshold_results.csv",
        "task4_regression_results.csv",
        "final_test_results.csv",
    ]:
        print(f"plots/{name}")
    print("\nBest tuned model:", best_tuned["Model"])
    print("\nFinal Test ROC-AUC:", round(final_auc, 4))


if __name__ == "__main__":
    main()
