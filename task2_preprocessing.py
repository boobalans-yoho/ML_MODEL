from pathlib import Path
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR.mkdir(exist_ok=True)
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


def create_popularity_band(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.copy()

    q1 = series.quantile(0.33)
    q2 = series.quantile(0.66)

    def assign(value):
        if pd.isna(value):
            return np.nan
        if value <= q1:
            return "Low"
        if value <= q2:
            return "Medium"
        return "High"

    return series.apply(assign)


FILE_PATH = find_dataset()
df = pd.read_csv(FILE_PATH)

print("=" * 70)
print("TASK 2 - SPOTIFY PREPROCESSING")
print("=" * 70)
print("\nDataset Shape:")
print(df.shape)
print("\nFirst 5 Rows:")
print(df.head())

TARGET = "is_dance_hit"
if TARGET not in df.columns:
    raise ValueError(f"'{TARGET}' target column not found.")

print("\nTarget:", TARGET)
print("\nTarget Distribution:")
print(df[TARGET].value_counts())

X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].astype(int)

genre_columns = [column for column in X.columns if column.startswith("track_genre_")]
X = X.drop(columns=genre_columns + ["track_genre", "track_genre_le"], errors="ignore")
print("\nGenre columns removed:", len(genre_columns))

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

if {"energy", "loudness"}.issubset(X.columns):
    X["energy_loudness_interaction"] = X["energy"] * X["loudness"]
    print("Created: energy_loudness_interaction")

if {"tempo", "energy"}.issubset(X.columns):
    X["tempo_energy_ratio"] = X["tempo"] / X["energy"].replace(0, np.nan)
    print("Created: tempo_energy_ratio")

if "duration_log" not in X.columns:
    if "duration_ms" in X.columns:
        X["log_duration"] = np.log1p(X["duration_ms"].clip(lower=0))
        print("Created: log_duration")
    else:
        print("duration_ms not available.")
else:
    X["log_duration"] = X["duration_log"]
    print("Created: log_duration from duration_log")

if "popularity" in X.columns:
    X["popularity_band"] = create_popularity_band(X["popularity"])
    print("Created ordered feature: popularity_band")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
ordinal_features = []
if "popularity_band" in categorical_features:
    categorical_features.remove("popularity_band")
    ordinal_features.append("popularity_band")

print("\n" + "=" * 70)
print("FEATURE TYPES")
print("=" * 70)
print("\nNumeric Features:")
print(numeric_features)
print("\nNominal Categorical Features:")
print(categorical_features)
print("\nOrdinal Features:")
print(ordinal_features)

numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

ordinal_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ordinal",
            OrdinalEncoder(
                categories=[["Low", "Medium", "High"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ]
)

transformers = []
if len(numeric_features) > 0:
    transformers.append(("numeric", numeric_pipeline, numeric_features))
if len(categorical_features) > 0:
    transformers.append(("categorical", categorical_pipeline, categorical_features))
if len(ordinal_features) > 0:
    transformers.append(("ordinal", ordinal_pipeline, ordinal_features))

preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

model_pipeline = Pipeline(
    [
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

print("\n" + "=" * 70)
print("PIPELINE")
print("=" * 70)
print(model_pipeline)

print("\n" + "=" * 70)
print("PIPELINE CROSS VALIDATION")
print("=" * 70)
cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring="accuracy")
for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i}: {score:.4f}")
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()
print(f"\nPipeline CV Accuracy: {cv_mean:.4f}")
print(f"Pipeline CV Standard Deviation: {cv_std:.4f}")

print("\n" + "=" * 70)
print("FITTING PIPELINE")
print("=" * 70)
model_pipeline.fit(X_train, y_train)
print("Pipeline fitted successfully.")

test_accuracy = model_pipeline.score(X_test, y_test)
print("\nTest Accuracy:", round(test_accuracy, 4))

MODEL_PATH = MODELS_DIR / "spotify_pipeline.joblib"
joblib.dump(model_pipeline, MODEL_PATH)
print("\nPipeline saved to:")
print(MODEL_PATH)

print("\n" + "=" * 70)
print("LOADING SAVED PIPELINE")
print("=" * 70)
loaded_pipeline = joblib.load(MODEL_PATH)
print("Pipeline loaded successfully.")

print("\n" + "=" * 70)
print("PREDICTION FOR 3 NEW RAW ROWS")
print("=" * 70)
new_rows = df.drop(columns=[TARGET]).iloc[:3].copy()
new_rows = new_rows.drop(columns=genre_columns + ["track_genre", "track_genre_le"], errors="ignore")

if {"energy", "loudness"}.issubset(new_rows.columns):
    new_rows["energy_loudness_interaction"] = new_rows["energy"] * new_rows["loudness"]
if {"tempo", "energy"}.issubset(new_rows.columns):
    new_rows["tempo_energy_ratio"] = new_rows["tempo"] / new_rows["energy"].replace(0, np.nan)
if "duration_log" not in new_rows.columns:
    if "duration_ms" in new_rows.columns:
        new_rows["log_duration"] = np.log1p(new_rows["duration_ms"].clip(lower=0))
elif "duration_log" in new_rows.columns:
    new_rows["log_duration"] = new_rows["duration_log"]

if "popularity" in new_rows.columns:
    popularity_q1 = X_train["popularity"].quantile(0.33)
    popularity_q2 = X_train["popularity"].quantile(0.66)

    def new_popularity_category(value):
        if pd.isna(value):
            return np.nan
        if value <= popularity_q1:
            return "Low"
        if value <= popularity_q2:
            return "Medium"
        return "High"

    new_rows["popularity_band"] = new_rows["popularity"].apply(new_popularity_category)

predictions = loaded_pipeline.predict(new_rows)
probabilities = loaded_pipeline.predict_proba(new_rows)

print("\nPredictions for 3 new rows:")
print(predictions)
print("\nPrediction probabilities:")
print(probabilities)

cv_result = pd.DataFrame({"Fold": [1, 2, 3, 4, 5], "Accuracy": cv_scores})
cv_result.to_csv(PLOTS_DIR / "task2_cv_accuracy.csv", index=False)

print("\n" + "=" * 70)
print("TASK 2 COMPLETED")
print("=" * 70)
print("\nUsed:")
print("1. SimpleImputer - median for numeric values")
print("2. SimpleImputer - most frequent for categorical values")
print("3. OneHotEncoder - nominal categorical features")
print("4. OrdinalEncoder - popularity Low < Medium < High")
print("5. StandardScaler - numeric features")
print("6. Feature engineering")
print("7. ColumnTransformer")
print("8. Pipeline")
print("9. 5-fold Cross Validation")
print("10. Joblib model saving/loading")
