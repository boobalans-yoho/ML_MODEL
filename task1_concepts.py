import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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


FILE_PATH = find_dataset()

df = pd.read_csv(FILE_PATH)

print("=" * 70)
print("SPOTIFY DATASET")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

TARGET = "is_dance_hit"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found in the dataset.")

print("\nTarget Column:")
print(TARGET)

print("\nTarget Distribution:")
print(df[TARGET].value_counts())

X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].astype(int)

leakage_columns = []
for column in X.columns:
    if column == "track_genre":
        leakage_columns.append(column)
    elif column.startswith("track_genre_"):
        leakage_columns.append(column)
    elif column == "track_genre_le":
        leakage_columns.append(column)

X = X.drop(columns=leakage_columns, errors="ignore")

print("\nRemoved Leakage Columns:")
print(leakage_columns)

print("\nRemaining Feature Count:")
print(X.shape[1])

numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
X = X[numeric_columns]

print("\nNumeric Feature Count:")
print(len(numeric_columns))

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

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)
print("\nApproximate percentages:")
print("Train:", len(X_train) / len(df) * 100, "%")
print("Validation:", len(X_val) / len(df) * 100, "%")
print("Test:", len(X_test) / len(df) * 100, "%")

print("\n" + "=" * 70)
print("DECISION TREE OVERFITTING")
print("=" * 70)

depths = [1, 3, 5, 10, None]
train_scores = []
validation_scores = []

for depth in depths:
    imputer = SimpleImputer(strategy="median")
    X_train_processed = imputer.fit_transform(X_train)
    X_val_processed = imputer.transform(X_val)

    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train_processed, y_train)

    train_prediction = model.predict(X_train_processed)
    validation_prediction = model.predict(X_val_processed)

    train_accuracy = accuracy_score(y_train, train_prediction)
    validation_accuracy = accuracy_score(y_val, validation_prediction)

    train_scores.append(train_accuracy)
    validation_scores.append(validation_accuracy)

    print(
        f"Depth {depth}: "
        f"Train Accuracy = {train_accuracy:.4f}, "
        f"Validation Accuracy = {validation_accuracy:.4f}"
    )

depth_labels = ["1", "3", "5", "10", "None"]

plt.figure(figsize=(9, 6))
plt.plot(depth_labels, train_scores, marker="o", label="Train Accuracy")
plt.plot(depth_labels, validation_scores, marker="o", label="Validation Accuracy")
plt.xlabel("Decision Tree Max Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree - Overfitting Analysis")
plt.legend()
plt.grid(True)
plt.savefig(PLOTS_DIR / "task1_tree_overfitting.png", dpi=300, bbox_inches="tight")
if matplotlib.get_backend().lower() != "agg":
    plt.show()
else:
    plt.close()

print("\n" + "=" * 70)
print("5-FOLD CROSS VALIDATION - LOGISTIC REGRESSION")
print("=" * 70)

logistic_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

cv_scores = cross_val_score(
    logistic_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
)

print("\nIndividual CV Scores:")
for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i}: {score:.4f}")

cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

print(f"\nCV Mean: {cv_mean:.4f}")
print(f"CV Standard Deviation: {cv_std:.4f}")
print(f"\nCV Result: {cv_mean:.4f} +/- {cv_std:.4f}")

print("\n" + "=" * 70)
print("DATA LEAKAGE DEMONSTRATION")
print("=" * 70)

print("\nWRONG APPROACH:")
print("Scaler is fitted before the train/validation separation.")

scaler_wrong = StandardScaler()
combined_data = pd.concat([X_train, X_val])
combined_imputer = SimpleImputer(strategy="median")
combined_imputed = combined_imputer.fit_transform(combined_data)

scaler_wrong.fit(combined_imputed)

X_train_wrong = scaler_wrong.transform(combined_imputer.transform(X_train))
X_val_wrong = scaler_wrong.transform(combined_imputer.transform(X_val))

wrong_model = LogisticRegression(max_iter=1000, random_state=42)
wrong_model.fit(X_train_wrong, y_train)
wrong_prediction = wrong_model.predict(X_val_wrong)
wrong_accuracy = accuracy_score(y_val, wrong_prediction)

print("Wrong validation accuracy:", round(wrong_accuracy, 4))

print("\nCORRECT APPROACH:")
print("Scaler is fitted only using training data.")

correct_imputer = SimpleImputer(strategy="median")
X_train_imputed = correct_imputer.fit_transform(X_train)
X_val_imputed = correct_imputer.transform(X_val)

scaler_correct = StandardScaler()
X_train_correct = scaler_correct.fit_transform(X_train_imputed)
X_val_correct = scaler_correct.transform(X_val_imputed)

correct_model = LogisticRegression(max_iter=1000, random_state=42)
correct_model.fit(X_train_correct, y_train)
correct_prediction = correct_model.predict(X_val_correct)
correct_accuracy = accuracy_score(y_val, correct_prediction)

print("Correct validation accuracy:", round(correct_accuracy, 4))

print("\n" + "=" * 70)
print("DATA LEAKAGE EXPLANATION")
print("=" * 70)

print(
    "1. Data leakage happens when information from validation/test "
    "data reaches the training process."
)
print(
    "2. In the wrong approach, the scaler learns statistics from "
    "both training and validation data."
)
print(
    "3. In the correct approach, the scaler is fitted only on "
    "training data."
)
print(
    "4. Validation and test data are transformed using the "
    "already-fitted training scaler."
)

print("\n" + "=" * 70)
print("SUPERVISED VS UNSUPERVISED")
print("=" * 70)

problems = {
    "Spam detection": "Supervised - data contains known spam/not-spam labels.",
    "Customer segmentation": "Unsupervised - groups customers without predefined labels.",
    "House price prediction": "Supervised - known house prices are used as targets.",
    "Anomaly detection": "Usually unsupervised - the model finds unusual observations.",
}

for problem, explanation in problems.items():
    print(f"\n{problem}:")
    print(explanation)

print("\n" + "=" * 70)
print("7-STEP MACHINE LEARNING WORKFLOW")
print("=" * 70)

workflow = [
    "1. Collect the dataset",
    "2. Understand and explore the data",
    "3. Clean and preprocess the data",
    "4. Engineer and select features",
    "5. Split data into train, validation and test",
    "6. Train and evaluate machine learning models",
    "7. Tune, select and deploy the best model",
]

for step in workflow:
    print(step)

print("\n" + "=" * 70)
print("TASK 1 COMPLETED")
print("=" * 70)

print("\nPlot saved to:")
print("plots/task1_tree_overfitting.png")

print("\nTest set was kept separate and was NOT used for model selection.")
