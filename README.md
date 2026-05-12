# Bank Marketing Subscription Prediction

This repository contains a machine learning project and Streamlit dashboard for
predicting whether a bank customer will subscribe to a term deposit. The model
uses the UCI Bank Marketing dataset and a logistic regression pipeline
implemented from scratch with NumPy and pandas.

## Repository Contents

| File | Description |
| --- | --- |
| `app.py` | Main Streamlit application. This is the file to run for the interactive dashboard. |
| `MODEL1_FINAL_frontend.py` | Core machine learning script. It contains data loading, preprocessing, logistic regression, evaluation, threshold analysis, feature analysis, prediction helpers, and a command-line demo entry point. |
| `final model.ipynb` | Notebook version of the final modeling work and experiments. |
| `bank-full.csv` | Local Bank Marketing dataset used by the Streamlit app. |
| `README.md` | Project documentation and run instructions. |

Running the training pipeline may also generate these local artifacts:

| File | Description |
| --- | --- |
| `lr_model.pkl` | Saved logistic regression model. |
| `preprocessor.pkl` | Saved fitted preprocessing object. |

## Project Overview

The pipeline predicts the target variable `y`, where:

- `yes` means the customer subscribed to a term deposit
- `no` means the customer did not subscribe

The project includes:

- Data summary and class imbalance analysis
- Handling of `unknown` values
- Train/test split before preprocessing to avoid data leakage
- Mean imputation for numeric features
- Mode imputation for categorical features
- One-hot encoding for categorical variables
- Standardization for numeric variables
- Logistic regression implemented from scratch
- Accuracy, precision, recall, F1-score, and confusion matrix evaluation
- Threshold tuning for imbalanced classification
- Feature coefficient analysis
- Single-customer prediction through the Streamlit interface

## Install Dependencies

From the repository root, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install streamlit pandas numpy matplotlib ucimlrepo
```

## Run the Streamlit Application

Start the dashboard from the repository root:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser. The app trains the model once per session and
then uses the trained pipeline across the dashboard pages.

## Dashboard Pages

The sidebar menu includes:

- `Data Exploration`: dataset size, class distribution, numeric statistics, and unknown-value audit
- `Feature Description`: descriptions of the dataset features
- `Preprocessing`: preprocessing steps, imputation values, one-hot encoding summary, and scaling checks
- `Training`: learning rate, iteration count, final loss, test metrics, and training loss curve
- `Evaluation`: metrics, confusion matrices, and sample predictions at a selected threshold
- `Threshold Analysis`: accuracy, precision, recall, and F1-score across decision thresholds
- `Feature Analysis`: positive and negative logistic regression coefficients
- `Prediction`: form for scoring a single customer profile

## Run the Command-Line Pipeline

The core script can also be run directly:

```bash
python MODEL1_FINAL_frontend.py
```

This prints the full modeling workflow in the terminal, including preprocessing,
training, evaluation, threshold tuning, sensitivity experiments, feature
analysis, and artifact saving.

## Data Notes

- `app.py` first looks for `bank-full.csv` in the repository folder.
- If local data is unavailable, the pipeline code can fetch the UCI Bank
  Marketing dataset through `ucimlrepo`, which requires internet access.
- The target class is imbalanced, so threshold tuning is important. A lower
  threshold can improve recall, which is useful when missing potential
  subscribers is more costly than making extra calls.
- The `duration` feature should be interpreted carefully because it is only
  known after a phone call has occurred and may cause leakage in a real
  production prediction setting.
