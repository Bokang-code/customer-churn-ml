# Customer Churn Prediction

An end-to-end machine-learning project that explores customer churn in the telecommunications sector. The workflow loads and prepares customer data, trains an XGBoost classifier, evaluates its predictions, and records experiments and model artefacts with MLflow.

The project is structured as a small, modular ML pipeline, separating data loading, preprocessing and model training.

## Project overview

Customer churn occurs when a customer stops using a service. Identifying customers who may be at risk of leaving can help organisations investigate retention opportunities and make more informed decisions.

This project uses the Telco Customer Churn dataset and the `Churn` field as its prediction target.

### Current workflow

1. **Load data** from a local CSV file.
2. **Prepare features** by cleaning column names, removing customer identifiers, converting `TotalCharges` to numeric, encoding categorical variables and handling missing numeric values.
3. **Train a classifier** using XGBoost with an 80/20 train-test split.
4. **Evaluate the model** using accuracy and recall.
5. **Track the experiment** in MLflow, logging parameters, metrics, the model and the input dataset.

## Tech stack

- **Python** — project language
- **Pandas / NumPy** — data handling
- **scikit-learn** — preprocessing utilities, train-test split and evaluation
- **XGBoost** — classification model
- **MLflow** — experiment tracking and model logging
- **Jupyter Notebook** — exploratory data analysis
- **pytest** — testing

## Repository structure

```text
.
├── data/
│   └── raw/                  # Place the source CSV here (not committed)
├── notebooks/
│   └── EDA.ipynb             # Exploratory data analysis
├── src/
│   ├── app/                  # Application components
│   ├── data/
│   │   ├── load_data.py      # CSV loading
│   │   └── preprocess.py     # Data cleaning and encoding
│   ├── features/             # Feature-related modules
│   ├── models/
│   │   └── train.py          # XGBoost training and MLflow logging
│   ├── serving/              # Model serving components
│   └── utils/                # Shared utilities
├── tests/                    # Automated tests
├── train_and_extract.py      # Main training and artefact-discovery script
└── requirements.txt
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Bokang-code/customer-churn-ml.git
cd customer-churn-ml
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset

Place the Telco Customer Churn CSV at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset is not included in this repository. Obtain it from an authorised source and check its usage terms before use.

### 5. Train the model

Run the script from the repository root:

```bash
python train_and_extract.py
```

The script loads and preprocesses the dataset, trains the XGBoost model, and reports the latest MLflow run and its artefact location.

### 6. Explore experiment results

After training, launch the MLflow UI from the repository root:

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

## Model and evaluation

The current training implementation uses an `XGBClassifier` with 300 estimators, a learning rate of 0.1 and a maximum tree depth of 6. The data is split into training and test sets using an 80/20 split and a fixed random seed.

Accuracy and recall are logged to MLflow. No performance figures are stated here because results depend on the dataset and the run configuration; run the pipeline to reproduce them.

## Notes and next steps

This repository represents a development project, not a production-ready churn system. Useful next improvements include:

- Use a stratified split and cross-validation to make evaluation more robust.
- Compare recall, precision, F1-score, ROC-AUC and a confusion matrix, particularly because churn datasets can be imbalanced.
- Fit categorical encoders on training data and reuse them for inference to avoid data leakage and ensure consistent transformations.
- Add a reproducible inference pipeline and model-serving instructions.
- Expand automated tests for preprocessing and training.
- Track dataset provenance and experiment configuration.

## Reproducibility

The training script expects the dataset at the path shown above. MLflow stores run information and artefacts locally by default; generated tracking files should remain outside version control. The repository's `.gitignore` excludes local environments, caches and MLflow run data.

---

**Author:** [Bokang Khasipe](https://github.com/Bokang-code)
