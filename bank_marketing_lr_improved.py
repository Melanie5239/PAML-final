"""
Logistic Regression Pipeline for UCI Bank Marketing Dataset
Implemented from scratch using only NumPy and Pandas.

IMPROVEMENTS:
- Train-test split BEFORE preprocessing (prevents data leakage)
- Preprocessor class with fit/transform API
- Confusion matrix output
- Threshold tuning capability
"""

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================

def load_data():
    """Fetch the Bank Marketing dataset from UCI ML Repository."""
    print("Loading dataset...")
    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y


def encode_target(y: pd.DataFrame) -> np.ndarray:
    """Convert 'yes'/'no' target labels to 1/0."""
    col = y.iloc[:, 0]
    return col.map({"yes": 1, "no": 0}).to_numpy(dtype=np.float64)


# =============================================================================
# SECTION 2: TRAIN-TEST SPLIT (BEFORE PREPROCESSING)
# =============================================================================

def train_test_split_df(X: pd.DataFrame, y: pd.DataFrame,
                         test_size: float = 0.2,
                         random_seed: int = 42):
    """
    Shuffle and split DataFrames into train/test sets.
    This happens BEFORE any preprocessing to prevent data leakage.
    """
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(len(X))

    split_idx = int(len(X) * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


# =============================================================================
# SECTION 3: PREPROCESSOR CLASS (FIT ON TRAIN, TRANSFORM ON TRAIN & TEST)
# =============================================================================

class Preprocessor:
    """
    Preprocessing pipeline that prevents data leakage by:
      1. Fitting all statistics (mean, mode, std) on training data only
      2. Applying the same transformations to both train and test sets

    Methods:
      fit(X_train): Learn preprocessing parameters from training data
      transform(X): Apply learned transformations to any dataset
      fit_transform(X_train): Convenience method combining fit + transform
    """

    def __init__(self):
        # Imputation parameters
        self.numerical_means_ = None
        self.categorical_modes_ = None
        self.numerical_cols_ = None
        self.categorical_cols_ = None

        # One-hot encoding parameters
        self.dummy_columns_ = None

        # Standardization parameters
        self.scaling_means_ = None
        self.scaling_stds_ = None

    # -------------------------------------------------------------------------
    # Step 1: Replace "unknown" with NaN
    # -------------------------------------------------------------------------

    @staticmethod
    def _replace_unknown_with_nan(X: pd.DataFrame) -> pd.DataFrame:
        """Replace 'unknown' string values with NaN."""
        return X.replace("unknown", np.nan)

    # -------------------------------------------------------------------------
    # Step 2: Imputation (fit on train, apply to train/test)
    # -------------------------------------------------------------------------

    def _fit_imputation(self, X: pd.DataFrame):
        """Learn imputation values from training data."""
        self.numerical_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Compute mean for numerical columns
        self.numerical_means_ = X[self.numerical_cols_].mean()

        # Compute mode for categorical columns
        self.categorical_modes_ = {}
        for col in self.categorical_cols_:
            mode_values = X[col].mode()
            self.categorical_modes_[col] = mode_values[0] if len(mode_values) > 0 else np.nan

    def _transform_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned imputation values."""
        X = X.copy()

        # Fill numerical columns with training means
        for col in self.numerical_cols_:
            if col in X.columns:
                X[col] = X[col].fillna(self.numerical_means_[col])

        # Fill categorical columns with training modes
        for col in self.categorical_cols_:
            if col in X.columns:
                X[col] = X[col].fillna(self.categorical_modes_[col])

        return X

    # -------------------------------------------------------------------------
    # Step 3: One-hot encoding (fit on train, align test to match)
    # -------------------------------------------------------------------------

    def _fit_one_hot_encoding(self, X: pd.DataFrame):
        """Learn dummy column names from training data."""
        X_encoded = pd.get_dummies(X, drop_first=False)
        self.dummy_columns_ = X_encoded.columns.tolist()

    def _transform_one_hot_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot encoding and ensure columns match training set.
        Missing columns in test set are filled with 0.
        Extra columns in test set are dropped.
        """
        X_encoded = pd.get_dummies(X, drop_first=False)

        # Add missing columns (fill with 0)
        for col in self.dummy_columns_:
            if col not in X_encoded.columns:
                X_encoded[col] = 0

        # Remove extra columns not seen in training
        extra_cols = set(X_encoded.columns) - set(self.dummy_columns_)
        X_encoded = X_encoded.drop(columns=list(extra_cols))

        # Reorder columns to match training set
        X_encoded = X_encoded[self.dummy_columns_]

        return X_encoded

    # -------------------------------------------------------------------------
    # Step 4: Standardization (fit on train, apply to train/test)
    # -------------------------------------------------------------------------

    def _fit_standardization(self, X: pd.DataFrame):
        """Learn mean and std from training data (after one-hot encoding)."""
        num_cols = X.select_dtypes(include=[np.number]).columns
        self.scaling_means_ = X[num_cols].mean()
        self.scaling_stds_ = X[num_cols].std()

    def _transform_standardization(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score standardization using training statistics."""
        X = X.copy()
        num_cols = X.select_dtypes(include=[np.number]).columns

        # Avoid division by zero for constant columns
        stds_safe = self.scaling_stds_.replace(0, 1)

        X[num_cols] = (X[num_cols] - self.scaling_means_) / stds_safe

        return X

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame):
        """
        Fit the preprocessor on training data.
        Learns all parameters needed for transformation.
        """
        print("  Fitting preprocessor on training data...")

        # Step 1: Replace unknown
        X = self._replace_unknown_with_nan(X_train)

        # Step 2: Fit imputation
        self._fit_imputation(X)
        X = self._transform_imputation(X)

        # Step 3: Fit one-hot encoding
        self._fit_one_hot_encoding(X)
        X = self._transform_one_hot_encoding(X)

        # Step 4: Fit standardization
        self._fit_standardization(X)

        print(f"  Preprocessor fitted: {len(self.dummy_columns_)} features after encoding.")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted parameters.
        Can be applied to both training and test sets.
        """
        # Step 1: Replace unknown
        X = self._replace_unknown_with_nan(X)

        # Step 2: Apply imputation
        X = self._transform_imputation(X)

        # Step 3: Apply one-hot encoding
        X = self._transform_one_hot_encoding(X)

        # Step 4: Apply standardization
        X = self._transform_standardization(X)

        return X.to_numpy(dtype=np.float64)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Convenience method: fit on training data and transform it."""
        self.fit(X_train)
        return self.transform(X_train)


# =============================================================================
# SECTION 4: LOGISTIC REGRESSION (FROM SCRATCH - UNCHANGED)
# =============================================================================

class LogisticRegressionScratch:
    """
    Binary Logistic Regression trained with batch gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent updates.
    n_iterations : int
        Number of full passes over the training data.
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation: σ(z) = 1 / (1 + e^{-z}).
        Clips z to [-500, 500] to prevent numerical overflow.
        """
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def _binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Binary cross-entropy loss:
          L = -1/n * Σ [ y*log(p) + (1-y)*log(1-p) ]
        """
        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return -np.mean(
            y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model using batch gradient descent.

        Gradient derivations (from BCE loss):
          dL/dw = (1/n) * Xᵀ · (ŷ - y)
          dL/db = (1/n) * Σ(ŷ - y)
        """
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iterations):
            linear_output = X @ self.weights + self.bias
            y_pred = self._sigmoid(linear_output)

            loss = self._binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

            error = y_pred - y
            dw = (X.T @ error) / n_samples
            db = np.mean(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if (i + 1) % 100 == 0:
                print(f"  Iteration {i + 1:>5}/{self.n_iterations} | Loss: {loss:.6f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities P(y=1 | X)."""
        linear_output = X @ self.weights + self.bias
        return self._sigmoid(linear_output)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary class predictions using the given threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)


# =============================================================================
# SECTION 5: EVALUATION METRICS (FROM SCRATCH)
# =============================================================================

def confusion_matrix_values(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute the four entries of the confusion matrix.
    Returns: (TP, FP, TN, FN)
    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, tn, fn


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy = (TP + TN) / total"""
    return float(np.mean(y_true == y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision = TP / (TP + FP)"""
    tp, fp, _, _ = confusion_matrix_values(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall = TP / (TP + FN)"""
    tp, _, _, fn = confusion_matrix_values(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1-Score = 2 * (Precision * Recall) / (Precision + Recall)"""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all metrics and return them as a dictionary."""
    return {
        "Accuracy":  accuracy(y_true, y_pred),
        "Precision": precision(y_true, y_pred),
        "Recall":    recall(y_true, y_pred),
        "F1-Score":  f1_score(y_true, y_pred),
    }


# =============================================================================
# SECTION 6: RESULTS OUTPUT
# =============================================================================

def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix"):
    """Print confusion matrix in a readable format."""
    tp, fp, tn, fn = confusion_matrix_values(y_true, y_pred)

    print(f"\n{title}")
    print("=" * 40)
    print(f"  True Positives  (TP): {tp:>6}")
    print(f"  False Positives (FP): {fp:>6}")
    print(f"  True Negatives  (TN): {tn:>6}")
    print(f"  False Negatives (FN): {fn:>6}")
    print("=" * 40)


def print_results(metrics: dict, split_name: str = "Test"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 40}")
    print(f"  Evaluation Results ({split_name} Set)")
    print(f"{'=' * 40}")
    for metric, value in metrics.items():
        print(f"  {metric:<12}: {value:.4f}")
    print(f"{'=' * 40}")


def print_threshold_comparison(model, X: np.ndarray, y: np.ndarray,
                                thresholds: list, split_name: str = "Test"):
    """Compare metrics across different classification thresholds."""
    print(f"\n{'=' * 60}")
    print(f"  Threshold Comparison ({split_name} Set)")
    print(f"{'=' * 60}")
    print(f"  {'Threshold':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print(f"  {'-' * 58}")

    for threshold in thresholds:
        y_pred = model.predict(X, threshold=threshold)
        metrics = evaluate(y, y_pred)
        print(f"  {threshold:<12.2f} "
              f"{metrics['Accuracy']:<10.4f} "
              f"{metrics['Precision']:<10.4f} "
              f"{metrics['Recall']:<10.4f} "
              f"{metrics['F1-Score']:<10.4f}")

    print(f"{'=' * 60}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  IMPROVED LOGISTIC REGRESSION PIPELINE")
    print("  (No Data Leakage - Split Before Preprocessing)")
    print("=" * 60 + "\n")

    # -------------------------------------------------------------------------
    # Step 1: Load raw data
    # -------------------------------------------------------------------------
    X_raw, y_raw = load_data()

    # -------------------------------------------------------------------------
    # Step 2: Train-test split FIRST (before any preprocessing)
    # -------------------------------------------------------------------------
    print("\nSplitting data (BEFORE preprocessing to prevent leakage)...")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split_df(
        X_raw, y_raw, test_size=0.2, random_seed=42
    )
    print(f"  Train samples: {X_train_raw.shape[0]}")
    print(f"  Test samples:  {X_test_raw.shape[0]}")

    # Encode target variables
    y_train = encode_target(y_train_raw)
    y_test = encode_target(y_test_raw)

    # -------------------------------------------------------------------------
    # Step 3: Fit preprocessor on training data ONLY
    # -------------------------------------------------------------------------
    print("\nPreprocessing:")
    preprocessor = Preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)

    # Apply same transformations to test data
    print("  Transforming test data using training parameters...")
    X_test = preprocessor.transform(X_test_raw)

    print(f"\n  Final feature count: {X_train.shape[1]}")

    # -------------------------------------------------------------------------
    # Step 4: Train Logistic Regression
    # -------------------------------------------------------------------------
    print("\nTraining Logistic Regression...")
    model = LogisticRegressionScratch(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # Step 5: Evaluate with default threshold (0.5)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  EVALUATION WITH DEFAULT THRESHOLD (0.5)")
    print("=" * 60)

    y_pred_train = model.predict(X_train, threshold=0.5)
    y_pred_test = model.predict(X_test, threshold=0.5)

    train_metrics = evaluate(y_train, y_pred_train)
    test_metrics = evaluate(y_test, y_pred_test)

    print_results(train_metrics, split_name="Train")
    print_confusion_matrix(y_train, y_pred_train, title="Confusion Matrix (Train Set)")

    print_results(test_metrics, split_name="Test")
    print_confusion_matrix(y_test, y_pred_test, title="Confusion Matrix (Test Set)")

    # -------------------------------------------------------------------------
    # Step 6: Threshold tuning analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  THRESHOLD TUNING ANALYSIS")
    print("=" * 60)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print_threshold_comparison(model, X_test, y_test, thresholds, split_name="Test")

    # -------------------------------------------------------------------------
    # Step 7: Detailed analysis with threshold=0.3
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  DETAILED EVALUATION WITH THRESHOLD = 0.3")
    print("  (Lower threshold to improve recall for imbalanced data)")
    print("=" * 60)

    y_pred_test_03 = model.predict(X_test, threshold=0.3)
    test_metrics_03 = evaluate(y_test, y_pred_test_03)

    print_results(test_metrics_03, split_name="Test (threshold=0.3)")
    print_confusion_matrix(y_test, y_pred_test_03, title="Confusion Matrix (Test, threshold=0.3)")

    # -------------------------------------------------------------------------
    # Step 8: Sample predictions
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SAMPLE PREDICTIONS (first 10 test samples)")
    print("=" * 60)

    y_proba_test = model.predict_proba(X_test)
    y_pred_05 = model.predict(X_test, threshold=0.5)
    y_pred_03 = model.predict(X_test, threshold=0.3)

    print(f"  {'#':<4} {'P(y=1)':<10} {'Pred@0.5':<10} {'Pred@0.3':<10} {'Actual':<8}")
    print(f"  {'-' * 46}")
    for i in range(10):
        print(f"  {i+1:<4} {y_proba_test[i]:<10.4f} {y_pred_05[i]:<10} "
              f"{y_pred_03[i]:<10} {int(y_test[i]):<8}")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
