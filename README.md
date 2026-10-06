# ML Assessment Project

This project analyzes a Spotify dataset for classification and regression tasks using scikit-learn, XGBoost, and plotting utilities.

## Project structure

- `task1_concepts.py` — exploratory data analysis and model-concept demonstrations
- `task2_preprocessing.py` — feature engineering and preprocessing pipeline
- `task3_models.py` — classification, regression, and clustering experiments
- `task4_evaluation.py` — model evaluation, threshold tuning, and hyperparameter search
- `data/` — dataset files
- `models/` — saved model artifacts
- `plots/` — generated charts and CSV results

## Environment

The project uses a local virtual environment in `.venv`.

To run scripts from the project root:

```bash
python task1_concepts.py
python task2_preprocessing.py
python task3_models.py
python task4_evaluation.py
```

If you are using the included virtual environment on Windows:

```powershell
.venv\Scripts\python.exe task1_concepts.py
.venv\Scripts\python.exe task2_preprocessing.py
.venv\Scripts\python.exe task3_models.py
.venv\Scripts\python.exe task4_evaluation.py
```

## Dataset

The main dataset is:

- `data/modified-spotify-dataset (1).csv`

The script automatically searches for the available CSV file in the `data/` folder.

## Main objectives

1. Explore the Spotify dataset and understand the target column.
2. Remove leakage columns and prepare valid features.
3. Train and compare classification models.
4. Evaluate model quality using accuracy, precision, recall, F1-score, and ROC-AUC.
5. Run regression modeling for popularity prediction.
6. Save plots and CSV summaries in the `plots/` folder.

## Notes

- The project uses `matplotlib` in non-interactive mode (`Agg`) to allow scripts to run headlessly.
- Model outputs are saved to the `plots/` directory for review and reporting.
