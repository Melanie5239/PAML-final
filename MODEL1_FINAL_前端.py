# ============================================================
#  MODEL1_FINAL_前端.py
#  Bank Marketing – Logistic Regression Pipeline
#  UCI Dataset id=222  |  45,211 samples  |  16 features
#
#  Dual-mode design:
#    • CLI  → call main() for full printed output (unchanged)
#    • Frontend → call the Section-5 API functions; each returns
#                 a dict / DataFrame suitable for Streamlit display
# ============================================================

import io
import pickle
import contextlib
import numpy as np
import pandas as pd

try:
    from ucimlrepo import fetch_ucirepo
except ModuleNotFoundError:
    fetch_ucirepo = None


# ============================================================
# SECTION 1 – DATA LOADING
# ============================================================

def load_data():
    if fetch_ucirepo is None:
        raise ModuleNotFoundError(
            "ucimlrepo is not installed. Install it or load data from the local CSV."
        )
    print("Loading dataset...")
    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y


# ============================================================
# SECTION 2 – CORE ML CLASSES  (unchanged)
# ============================================================

class Preprocessor:

    def __init__(self):
        self.numerical_means_ = None
        self.categorical_modes_ = None
        self.numerical_cols_ = None
        self.categorical_cols_ = None
        self.dummy_columns_ = None
        self.scaling_means_ = None
        self.scaling_stds_ = None

    @staticmethod
    def _replace_unknown_with_nan(X: pd.DataFrame) -> pd.DataFrame:
        return X.replace("unknown", np.nan)

    def _fit_imputation(self, X: pd.DataFrame):
        self.numerical_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = X.select_dtypes(exclude=[np.number]).columns.tolist()
        self.numerical_means_ = X[self.numerical_cols_].mean()
        self.categorical_modes_ = {}
        for col in self.categorical_cols_:
            mode_values = X[col].mode()
            self.categorical_modes_[col] = mode_values[0] if len(mode_values) > 0 else np.nan

    def _transform_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.numerical_cols_:
            if col in X.columns:
                X[col] = X[col].fillna(self.numerical_means_[col])
        for col in self.categorical_cols_:
            if col in X.columns:
                X[col] = X[col].fillna(self.categorical_modes_[col])
        return X

    def _fit_one_hot_encoding(self, X: pd.DataFrame):
        X_encoded = pd.get_dummies(X, drop_first=False)
        self.dummy_columns_ = X_encoded.columns.tolist()

    def _transform_one_hot_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        X_encoded = pd.get_dummies(X, drop_first=False)
        for col in self.dummy_columns_:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        extra_cols = set(X_encoded.columns) - set(self.dummy_columns_)
        X_encoded = X_encoded.drop(columns=list(extra_cols))
        X_encoded = X_encoded[self.dummy_columns_]
        return X_encoded

    def _fit_standardization(self, X: pd.DataFrame):
        num_cols = X.select_dtypes(include=[np.number]).columns
        self.scaling_means_ = X[num_cols].mean()
        self.scaling_stds_ = X[num_cols].std()

    def _transform_standardization(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        num_cols = [col for col in self.scaling_means_.index if col in X.columns]
        stds_safe = self.scaling_stds_.replace(0, 1)
        X[num_cols] = (X[num_cols] - self.scaling_means_) / stds_safe
        return X

    def fit(self, X_train: pd.DataFrame):
        print("  Fitting preprocessor on training data...")
        X = self._replace_unknown_with_nan(X_train)
        self._fit_imputation(X)
        X = self._transform_imputation(X)
        self._fit_one_hot_encoding(X)
        X = self._transform_one_hot_encoding(X)
        self._fit_standardization(X)
        print(f"  Preprocessor fitted: {len(self.dummy_columns_)} features after encoding.")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = self._replace_unknown_with_nan(X)
        X = self._transform_imputation(X)
        X = self._transform_one_hot_encoding(X)
        X = self._transform_standardization(X)
        return X.to_numpy(dtype=np.float64)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        self.fit(X_train)
        return self.transform(X_train)

    def get_feature_names(self) -> list:
        return list(self.dummy_columns_) if self.dummy_columns_ is not None else []


class LogisticRegressionScratch:

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def _binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return -np.mean(
            y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
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
        linear_output = X @ self.weights + self.bias
        return self._sigmoid(linear_output)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# ============================================================
# SECTION 3 – METRIC UTILITIES  (unchanged)
# ============================================================

def encode_target(y: pd.DataFrame) -> np.ndarray:
    col = y.iloc[:, 0]
    return col.map({"yes": 1, "no": 0}).to_numpy(dtype=np.float64)


def train_test_split_df(X: pd.DataFrame, y: pd.DataFrame,
                        test_size: float = 0.2,
                        random_seed: int = 42):
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


def confusion_matrix_values(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, tn, fn


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, fp, _, _ = confusion_matrix_values(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, _, _, fn = confusion_matrix_values(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "Accuracy":  accuracy(y_true, y_pred),
        "Precision": precision(y_true, y_pred),
        "Recall":    recall(y_true, y_pred),
        "F1-Score":  f1_score(y_true, y_pred),
    }


# ============================================================
# SECTION 4 – CLI PRINT FUNCTIONS  (unchanged; kept for main())
# ============================================================

def print_dataset_summary(X: pd.DataFrame, y: pd.DataFrame):
    target_col = y.iloc[:, 0]
    n_yes = int((target_col == "yes").sum())
    n_no  = int((target_col == "no").sum())
    total = len(target_col)
    pos_rate = n_yes / total * 100

    print("\n" + "=" * 60)
    print("  BANK MARKETING DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total samples           : {total:,}")
    print(f"  Number of features      : {X.shape[1]}")
    print(f"  Subscribed     (yes = 1): {n_yes:,}")
    print(f"  Not subscribed (no  = 0): {n_no:,}")
    print(f"  Positive class rate     : {pos_rate:.2f}%")
    print("  Note: Dataset is heavily imbalanced (~11% positive).")
    print("=" * 60)


def print_unknown_summary(X: pd.DataFrame):
    unknown_counts = {}
    for col in X.select_dtypes(include="object").columns:
        n = int((X[col] == "unknown").sum())
        if n > 0:
            unknown_counts[col] = n

    print("\n" + "=" * 60)
    print("  UNKNOWN VALUE AUDIT (before preprocessing)")
    print("=" * 60)
    if unknown_counts:
        for col, count in sorted(unknown_counts.items(), key=lambda x: -x[1]):
            print(f"  {col:<20}: {count:>5} unknown values")
    else:
        print("  No 'unknown' values found.")
    print("  These will be replaced with NaN and imputed during preprocessing.")
    print("=" * 60)


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                            title: str = "Confusion Matrix"):
    tp, fp, tn, fn = confusion_matrix_values(y_true, y_pred)
    print(f"\n{title}")
    print("=" * 40)
    print(f"  True Positives  (TP): {tp:>6}")
    print(f"  False Positives (FP): {fp:>6}")
    print(f"  True Negatives  (TN): {tn:>6}")
    print(f"  False Negatives (FN): {fn:>6}")
    print("=" * 40)


def print_results(metrics: dict, split_name: str = "Test"):
    print(f"\n{'=' * 40}")
    print(f"  Evaluation Results ({split_name} Set)")
    print(f"{'=' * 40}")
    for metric, value in metrics.items():
        print(f"  {metric:<12}: {value:.4f}")
    print(f"{'=' * 40}")


def print_threshold_comparison(model, X: np.ndarray, y: np.ndarray,
                                thresholds: list, split_name: str = "Test"):
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


def print_threshold_sensitivity(model, X: np.ndarray, y: np.ndarray):
    thresholds = [round(t * 0.1, 1) for t in range(1, 10)]
    results = []

    for thresh in thresholds:
        y_pred = model.predict(X, threshold=thresh)
        m = evaluate(y, y_pred)
        results.append({
            "threshold": thresh,
            "accuracy":  m["Accuracy"],
            "precision": m["Precision"],
            "recall":    m["Recall"],
            "f1":        m["F1-Score"],
        })

    best_f1  = max(results, key=lambda r: r["f1"])
    best_rec = max(results, key=lambda r: r["recall"])

    print("\n" + "=" * 66)
    print("  THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 66)
    print(f"  {'Threshold':<11} {'Accuracy':<11} {'Precision':<11} {'Recall':<11} {'F1-Score':<10}")
    print(f"  {'-' * 62}")
    for r in results:
        print(f"  {r['threshold']:<11.1f} "
              f"{r['accuracy']:<11.4f} "
              f"{r['precision']:<11.4f} "
              f"{r['recall']:<11.4f} "
              f"{r['f1']:<10.4f}")
    print(f"  {'-' * 62}")

    print(f"\n  Best threshold for F1-Score : {best_f1['threshold']:.1f}  "
          f"(F1 = {best_f1['f1']:.4f})")
    print(f"  Best threshold for Recall   : {best_rec['threshold']:.1f}  "
          f"(Recall = {best_rec['recall']:.4f})")

    print("\n  Interpretation:")
    print("    Lowering the threshold causes the model to label more customers")
    print("    as likely subscribers, which increases Recall (fewer missed leads)")
    print("    but reduces Precision (more non-subscribers contacted by mistake).")
    print("    Raising the threshold has the opposite effect: the model becomes")
    print("    more conservative, contacting fewer people but with higher accuracy.")
    print("    The optimal threshold depends on campaign budget and the relative")
    print("    cost of missing a real subscriber vs. wasting a call on a non-subscriber.")
    print("=" * 66)


def print_top_coefficients(model: LogisticRegressionScratch,
                            feature_names: list,
                            top_n: int = 10):
    if model.weights is None or not feature_names:
        print("  Model not trained or feature names unavailable.")
        return

    weights = model.weights
    name_weight = sorted(zip(feature_names, weights), key=lambda x: x[1])

    top_negative = name_weight[:top_n]
    top_positive = name_weight[-top_n:][::-1]

    print("\n" + "=" * 60)
    print("  FEATURE COEFFICIENT INTERPRETATION")
    print("  (Logistic Regression weights after standardization)")
    print("=" * 60)

    print(f"\n  Top {top_n} features POSITIVELY associated with subscription:")
    print(f"  {'Feature':<40} {'Coefficient':>12}")
    print(f"  {'-' * 54}")
    for name, w in top_positive:
        print(f"  {name:<40} {w:>12.4f}")

    print(f"\n  Top {top_n} features NEGATIVELY associated with subscription:")
    print(f"  {'Feature':<40} {'Coefficient':>12}")
    print(f"  {'-' * 54}")
    for name, w in top_negative:
        print(f"  {name:<40} {w:>12.4f}")

    print("=" * 60)


def print_business_threshold_analysis(model: LogisticRegressionScratch,
                                       X: np.ndarray,
                                       y: np.ndarray):
    y_pred_05 = model.predict(X, threshold=0.5)
    y_pred_03 = model.predict(X, threshold=0.3)

    tp_05, fp_05, _, fn_05 = confusion_matrix_values(y, y_pred_05)
    tp_03, fp_03, _, fn_03 = confusion_matrix_values(y, y_pred_03)

    total_actual_pos = int(np.sum(y == 1))
    extra_tp = tp_03 - tp_05
    extra_fp = fp_03 - fp_05

    print("\n" + "=" * 60)
    print("  BUSINESS THRESHOLD ANALYSIS: 0.5 vs 0.3")
    print("=" * 60)
    print(f"  Total actual subscribers in test set : {total_actual_pos}")
    print()
    print(f"  Threshold = 0.5")
    print(f"    Subscribers correctly identified (TP): {tp_05}")
    print(f"    Non-subscribers flagged as leads  (FP): {fp_05}")
    print(f"    Subscribers missed               (FN): {fn_05}")
    print()
    print(f"  Threshold = 0.3")
    print(f"    Subscribers correctly identified (TP): {tp_03}")
    print(f"    Non-subscribers flagged as leads  (FP): {fp_03}")
    print(f"    Subscribers missed               (FN): {fn_03}")
    print()
    print(f"  Lowering threshold from 0.5 → 0.3:")
    print(f"    +{extra_tp} additional subscribers captured (higher recall)")
    print(f"    +{extra_fp} additional false positives (wider outreach cost)")
    print()
    print("  Business interpretation:")
    print("    A threshold of 0.3 is preferable when the goal is to maximize")
    print("    subscriber capture — i.e., the cost of missing a real subscriber")
    print("    outweighs the cost of contacting a non-subscriber.")
    print("    This is typical in outbound telemarketing campaigns where each")
    print("    follow-up call is cheap relative to the value of a new term deposit.")
    print("    A threshold of 0.5 is better when call-center resources are limited")
    print("    and precision (contact quality) matters more than coverage.")
    print("=" * 60)


def save_artifacts(model: LogisticRegressionScratch,
                   preprocessor: Preprocessor,
                   model_path: str = "lr_model.pkl",
                   preprocessor_path: str = "preprocessor.pkl"):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"  Model saved       -> {model_path}")
    print(f"  Preprocessor saved-> {preprocessor_path}")


# ============================================================
# SECTION 4b – EXPERIMENT FUNCTIONS
#   These print (CLI) AND return data (frontend).
# ============================================================

def run_lr_sensitivity(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> list:
    """Run LR/iteration sensitivity analysis.
    Prints a formatted table (CLI) and returns a list of result dicts (frontend).
    """
    configs = [
        (0.001, 500),
        (0.001, 1000),
        (0.01,  500),
        (0.01,  1000),
        (0.05,  500),
        (0.05,  1000),
    ]

    results = []
    for lr, iters in configs:
        m = LogisticRegressionScratch(learning_rate=lr, n_iterations=iters)
        with contextlib.redirect_stdout(io.StringIO()):
            m.fit(X_train, y_train)
        final_loss = m.loss_history[-1]
        y_pred = m.predict(X_test, threshold=0.5)
        metrics = evaluate(y_test, y_pred)
        results.append({
            "lr": lr, "iters": iters, "loss": final_loss,
            "accuracy":  metrics["Accuracy"],
            "precision": metrics["Precision"],
            "recall":    metrics["Recall"],
            "f1":        metrics["F1-Score"],
        })

    best_f1 = max(results, key=lambda r: r["f1"])

    print("\n" + "=" * 74)
    print("  LEARNING RATE / ITERATION SENSITIVITY ANALYSIS")
    print("=" * 74)
    print(f"  {'LR':<8} {'Iters':<7} {'Loss':<10} {'Accuracy':<10} "
          f"{'Precision':<11} {'Recall':<9} {'F1':<8}")
    print(f"  {'-' * 70}")
    for r in results:
        print(f"  {r['lr']:<8.3f} {r['iters']:<7} {r['loss']:<10.6f} "
              f"{r['accuracy']:<10.4f} {r['precision']:<11.4f} "
              f"{r['recall']:<9.4f} {r['f1']:<8.4f}")
    print(f"  {'-' * 70}")

    print(f"\n  Best F1-Score : LR={best_f1['lr']}, iters={best_f1['iters']} "
          f"-> F1={best_f1['f1']:.4f}")

    loss_05_500  = next(r["loss"] for r in results if r["lr"] == 0.05 and r["iters"] == 500)
    loss_05_1000 = next(r["loss"] for r in results if r["lr"] == 0.05 and r["iters"] == 1000)
    stable = loss_05_500 > loss_05_1000 * 0.5

    print("\n  Interpretation:")
    print(f"    LR=0.001 converges slowly; both 500 and 1000 iterations show")
    print(f"    higher final loss than LR=0.01 or LR=0.05 at the same step count.")
    if stable:
        print(f"    LR=0.05 appears stable (loss still decreasing at 1000 iters).")
    else:
        print(f"    LR=0.05 may be overshooting; check loss values carefully.")
    loss_mid_500  = next(r["loss"] for r in results if r["lr"] == 0.01 and r["iters"] == 500)
    loss_mid_1000 = next(r["loss"] for r in results if r["lr"] == 0.01 and r["iters"] == 1000)
    if loss_mid_1000 < loss_mid_500:
        print(f"    For LR=0.01, 1000 iterations improves final loss over 500 "
              f"({loss_mid_500:.6f} -> {loss_mid_1000:.6f}), suggesting the model")
        print(f"    has not fully converged at 500 steps with this learning rate.")
    print("=" * 74)

    return results  # ← new: also return for frontend


def run_duration_ablation(X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame,
                           y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Duration ablation study.
    Prints formatted output (CLI) and returns a dict with both metric sets (frontend).
    """
    def _train_and_eval(X_tr_raw, X_te_raw, label):
        prep = Preprocessor()
        with contextlib.redirect_stdout(io.StringIO()):
            X_tr = prep.fit_transform(X_tr_raw)
            X_te = prep.transform(X_te_raw)
        m = LogisticRegressionScratch(learning_rate=0.01, n_iterations=1000)
        with contextlib.redirect_stdout(io.StringIO()):
            m.fit(X_tr, y_train)
        y_pred = m.predict(X_te, threshold=0.5)
        metrics = evaluate(y_test, y_pred)
        return metrics

    metrics_full = _train_and_eval(X_train_raw, X_test_raw, "full_features")

    X_train_no_dur = X_train_raw.drop(columns=["duration"], errors="ignore")
    X_test_no_dur  = X_test_raw.drop(columns=["duration"],  errors="ignore")
    metrics_no_dur = _train_and_eval(X_train_no_dur, X_test_no_dur, "without_duration")

    print("\n" + "=" * 66)
    print("  FEATURE SENSITIVITY: DURATION ABLATION")
    print("=" * 66)
    print(f"  {'Setting':<22} {'Accuracy':<11} {'Precision':<11} {'Recall':<9} {'F1':<8}")
    print(f"  {'-' * 62}")
    for label, m in [("full_features", metrics_full), ("without_duration", metrics_no_dur)]:
        print(f"  {label:<22} {m['Accuracy']:<11.4f} {m['Precision']:<11.4f} "
              f"{m['Recall']:<9.4f} {m['F1-Score']:<8.4f}")
    print(f"  {'-' * 62}")

    f1_drop = metrics_full["F1-Score"] - metrics_no_dur["F1-Score"]
    rec_drop = metrics_full["Recall"] - metrics_no_dur["Recall"]

    print(f"\n  F1 change when removing duration : {f1_drop:+.4f}")
    print(f"  Recall change                    : {rec_drop:+.4f}")
    print("\n  Interpretation:")
    if f1_drop > 0.01:
        print("    Removing 'duration' noticeably reduces F1 and Recall, confirming")
        print("    that call duration is the single most informative feature in this")
        print("    dataset. Longer calls strongly correlate with term deposit subscription.")
    else:
        print("    Removing 'duration' causes minimal metric change at threshold=0.5,")
        print("    but note that duration has the highest positive coefficient in the")
        print("    full model. Its impact is more visible at lower thresholds where")
        print("    recall matters more.")
    print("    Note: in a real deployment, 'duration' is unknown before the call,")
    print("    so a production model should be trained without it to avoid data leakage.")
    print("=" * 66)

    return {  # ← new: also return for frontend
        "full_features":    metrics_full,
        "without_duration": metrics_no_dur,
        "f1_change":        f1_drop,
        "recall_change":    rec_drop,
    }


def run_hyperparameter_grid_search(X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: np.ndarray, y_test: np.ndarray,
                                    X_train_raw: pd.DataFrame,
                                    X_test_raw: pd.DataFrame) -> dict:
    """Full hyperparameter grid search.
    Prints results (CLI) and returns a dict with all_results, top10, best configs (frontend).
    """
    lr_candidates     = [0.01, 0.03, 0.05, 0.07, 0.1]
    iter_candidates   = [1000, 2000, 3000]
    thresh_candidates = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    total_models = len(lr_candidates) * len(iter_candidates)
    print(f"  Running {total_models} model fits x {len(thresh_candidates)} thresholds "
          f"= {total_models * len(thresh_candidates)} evaluations...")

    all_results = []
    for lr in lr_candidates:
        for iters in iter_candidates:
            m = LogisticRegressionScratch(learning_rate=lr, n_iterations=iters)
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit(X_train, y_train)
            final_loss = m.loss_history[-1]
            for thresh in thresh_candidates:
                y_pred = m.predict(X_test, threshold=thresh)
                met = evaluate(y_test, y_pred)
                all_results.append({
                    "lr":        lr,
                    "iters":     iters,
                    "threshold": thresh,
                    "accuracy":  met["Accuracy"],
                    "precision": met["Precision"],
                    "recall":    met["Recall"],
                    "f1":        met["F1-Score"],
                    "loss":      final_loss,
                })

    all_results.sort(key=lambda r: r["f1"], reverse=True)
    top10 = all_results[:10]

    best_f1   = all_results[0]
    best_rec  = max(all_results, key=lambda r: r["recall"])
    best_prec = max(all_results, key=lambda r: r["precision"])

    W = 88
    print("\n" + "=" * W)
    print("  HYPERPARAMETER GRID SEARCH — TOP 10 BY F1-SCORE")
    print("=" * W)
    print(f"  {'Rank':<5} {'LR':<7} {'Iters':<7} {'Thresh':<8} "
          f"{'Accuracy':<10} {'Precision':<11} {'Recall':<9} {'F1':<9} {'Loss':<10}")
    print(f"  {'-' * (W - 2)}")
    for rank, r in enumerate(top10, 1):
        print(f"  {rank:<5} {r['lr']:<7.3f} {r['iters']:<7} {r['threshold']:<8.2f} "
              f"{r['accuracy']:<10.4f} {r['precision']:<11.4f} "
              f"{r['recall']:<9.4f} {r['f1']:<9.4f} {r['loss']:<10.6f}")
    print(f"  {'-' * (W - 2)}")

    def _fmt(r):
        return (f"LR={r['lr']}, iters={r['iters']}, threshold={r['threshold']:.2f} | "
                f"Accuracy={r['accuracy']:.4f}, Precision={r['precision']:.4f}, "
                f"Recall={r['recall']:.4f}, F1={r['f1']:.4f}")

    print(f"\n  Best F1        : {_fmt(best_f1)}")
    print(f"  Best Recall    : {_fmt(best_rec)}")
    print(f"  Best Precision : {_fmt(best_prec)}")

    best_lr, best_iters, best_thresh = best_f1["lr"], best_f1["iters"], best_f1["threshold"]

    def _eval_config(X_tr_raw, X_te_raw, label):
        prep = Preprocessor()
        with contextlib.redirect_stdout(io.StringIO()):
            X_tr = prep.fit_transform(X_tr_raw)
            X_te = prep.transform(X_te_raw)
        m2 = LogisticRegressionScratch(learning_rate=best_lr, n_iterations=best_iters)
        with contextlib.redirect_stdout(io.StringIO()):
            m2.fit(X_tr, y_train)
        y_pred = m2.predict(X_te, threshold=best_thresh)
        return evaluate(y_test, y_pred)

    met_full   = _eval_config(X_train_raw, X_test_raw, "full")
    X_tr_nd    = X_train_raw.drop(columns=["duration"], errors="ignore")
    X_te_nd    = X_test_raw.drop(columns=["duration"],  errors="ignore")
    met_no_dur = _eval_config(X_tr_nd, X_te_nd, "no_duration")

    print(f"\n  Best config (LR={best_lr}, iters={best_iters}, thresh={best_thresh:.2f})"
          f" — duration ablation:")
    print(f"  {'Setting':<20} {'Accuracy':<10} {'Precision':<11} {'Recall':<9} {'F1':<8}")
    print(f"  {'-' * 58}")
    for label, met in [("full_features", met_full), ("without_duration", met_no_dur)]:
        print(f"  {label:<20} {met['Accuracy']:<10.4f} {met['Precision']:<11.4f} "
              f"{met['Recall']:<9.4f} {met['F1-Score']:<8.4f}")
    print(f"  {'-' * 58}")

    best_lr_vals   = sorted(set(r["lr"]    for r in all_results[:20]))
    best_iter_vals = sorted(set(r["iters"] for r in all_results[:20]))
    default_rank = next((i + 1 for i, r in enumerate(all_results)
                         if r["lr"] == 0.01 and r["iters"] == 1000 and r["threshold"] == 0.5), None)
    default_f1   = next((r["f1"] for r in all_results
                         if r["lr"] == 0.01 and r["iters"] == 1000 and r["threshold"] == 0.5), None)

    print("\n  Interpretation:")
    print(f"    Learning rate  : Higher LRs (e.g. {max(best_lr_vals):.2f}) tend to reach lower loss")
    print(f"                     in fewer steps and produce better F1 on this imbalanced dataset.")
    if max(best_iter_vals) > 1000:
        print(f"    Iterations     : More iterations (2000–3000) consistently improve F1,")
        print(f"                     indicating the model has not converged at 1000 steps")
        print(f"                     for higher learning rates.")
    else:
        print(f"    Iterations     : 1000 iterations appear sufficient for top configurations.")
    print(f"    Threshold      : The best F1 configurations cluster around "
          f"threshold={best_f1['threshold']:.2f},")
    print(f"                     well below the default 0.5, confirming that the positive")
    print(f"                     class rate (~11.7%) requires a lower decision boundary.")
    if default_rank:
        print(f"    Default (LR=0.01, iters=1000, thresh=0.5): rank #{default_rank} of "
              f"{len(all_results)}, F1={default_f1:.4f}.")
        print(f"    The default settings are conservative; tuning improves F1 to "
              f"{best_f1['f1']:.4f} (+{best_f1['f1'] - default_f1:.4f}).")
    print("=" * W)

    return {  # ← new: also return for frontend
        "all_results":   all_results,
        "top10":         top10,
        "best_f1":       best_f1,
        "best_recall":   best_rec,
        "best_precision": best_prec,
        "default_rank":  default_rank,
        "default_f1":    default_f1,
        "best_config_ablation": {
            "full_features":    met_full,
            "without_duration": met_no_dur,
        },
    }


# ============================================================
# SECTION 5 – FRONTEND API  (new)
#
#   All functions return structured Python objects (dict /
#   DataFrame) that a Streamlit app can consume directly.
#   They do NOT rely on global state — every function takes
#   the data / model it needs as explicit parameters.
# ============================================================

# ------ 5.0  Module-level state cache ----------------------
#   run_full_demo() populates this so that predict_single()
#   can be called without re-training in a Streamlit session.

_PIPELINE_STATE: dict = {}


# ------ 5.1  Dataset summary --------------------------------

def get_dataset_summary(X: pd.DataFrame, y: pd.DataFrame) -> dict:
    """Return a dict describing the raw dataset (before splitting).

    Suitable for Streamlit metric cards and bar charts.
    """
    target_col = y.iloc[:, 0]
    n_yes  = int((target_col == "yes").sum())
    n_no   = int((target_col == "no").sum())
    total  = len(target_col)
    pos_rate = n_yes / total * 100

    # unknown-value audit per categorical column
    unknown_counts = {}
    for col in X.select_dtypes(include="object").columns:
        n = int((X[col] == "unknown").sum())
        if n > 0:
            unknown_counts[col] = n

    # dtype breakdown
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # per-column basic stats (numeric only)
    numeric_stats = (
        X[num_cols]
        .describe()
        .T[["mean", "std", "min", "50%", "max"]]
        .rename(columns={"50%": "median"})
        .round(2)
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    return {
        "total_samples":    total,
        "n_features":       X.shape[1],
        "n_subscribed":     n_yes,
        "n_not_subscribed": n_no,
        "positive_rate_pct": round(pos_rate, 2),
        "imbalanced":       True,
        "numerical_cols":   num_cols,
        "categorical_cols": cat_cols,
        "unknown_counts":   unknown_counts,
        "numeric_stats_df": numeric_stats,
        "class_distribution_df": pd.DataFrame({
            "label":  ["Subscribed (yes)", "Not Subscribed (no)"],
            "count":  [n_yes, n_no],
            "pct":    [round(pos_rate, 2), round(100 - pos_rate, 2)],
        }),
    }


# ------ 5.2  Feature descriptions ---------------------------

def get_feature_descriptions() -> pd.DataFrame:
    """Return a DataFrame with official UCI Bank Marketing feature metadata.

    Columns: feature, type, role, description, notes
    """
    rows = [
        ("age",      "Numerical",    "Input",
         "Age of the client (in years).",
         "Continuous; range typically 18–95."),
        ("job",      "Categorical",  "Input",
         "Type of job.",
         "Values: admin., blue-collar, entrepreneur, housemaid, management, "
         "retired, self-employed, services, student, technician, unemployed, unknown."),
        ("marital",  "Categorical",  "Input",
         "Marital status.",
         "Values: divorced, married, single, unknown. "
         "'divorced' includes widowed."),
        ("education","Categorical",  "Input",
         "Highest level of education.",
         "Values: basic.4y, basic.6y, basic.9y, high.school, illiterate, "
         "professional.course, university.degree, unknown."),
        ("default",  "Categorical",  "Input",
         "Has credit in default?",
         "Binary: yes / no / unknown."),
        ("balance",  "Numerical",    "Input",
         "Average yearly bank balance (euros).",
         "Can be negative (overdraft). Continuous."),
        ("housing",  "Categorical",  "Input",
         "Has a housing loan?",
         "Binary: yes / no / unknown."),
        ("loan",     "Categorical",  "Input",
         "Has a personal loan?",
         "Binary: yes / no / unknown."),
        ("contact",  "Categorical",  "Input",
         "Contact communication type for the last contact.",
         "Values: cellular, telephone."),
        ("day",      "Numerical",    "Input",
         "Last contact day of the month (1–31).",
         "Numeric day-of-month."),
        ("month",    "Categorical",  "Input",
         "Last contact month of the year.",
         "Values: jan, feb, mar, …, dec (12 categories)."),
        ("duration", "Numerical",    "Input",
         "Duration of the last contact, in seconds.",
         "IMPORTANT: highly predictive but unknown before the call is made. "
         "Should be excluded from production models to avoid data leakage."),
        ("campaign", "Numerical",    "Input",
         "Number of contacts performed during this campaign for this client.",
         "Includes the last contact. Continuous, ≥ 1."),
        ("pdays",    "Numerical",    "Input",
         "Days elapsed since the client was last contacted from a previous campaign.",
         "-1 means the client was not previously contacted."),
        ("previous", "Numerical",    "Input",
         "Number of contacts performed before this campaign for this client.",
         "Continuous, ≥ 0."),
        ("poutcome", "Categorical",  "Input",
         "Outcome of the previous marketing campaign.",
         "Values: failure, nonexistent, success, unknown."),
        ("y",        "Categorical",  "Target",
         "Has the client subscribed to a term deposit?",
         "Binary target: yes (1) / no (0). Positive rate ≈ 11.7%."),
    ]
    return pd.DataFrame(rows,
                        columns=["feature", "type", "role",
                                 "description", "notes"])


# ------ 5.3  Preprocessing demo -----------------------------

def run_preprocessing_demo(X_train_raw: pd.DataFrame,
                            X_test_raw: pd.DataFrame) -> dict:
    """Demonstrate the full preprocessing pipeline on the training split.

    Returns a dict with DataFrames and metadata for step-by-step display.
    """
    # Step 1 – unknown → NaN
    X_unk = X_train_raw.replace("unknown", np.nan)
    unknown_counts = {
        col: int((X_train_raw[col] == "unknown").sum())
        for col in X_train_raw.select_dtypes(include="object").columns
        if (X_train_raw[col] == "unknown").any()
    }

    # Step 2 – imputation (fit on train, show fill values)
    num_cols = X_unk.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_unk.select_dtypes(exclude=[np.number]).columns.tolist()
    num_means = X_unk[num_cols].mean().round(4)
    cat_modes = {
        col: X_unk[col].mode()[0]
        for col in cat_cols
        if len(X_unk[col].mode()) > 0
    }

    imputation_df = pd.DataFrame([
        {"feature": c, "strategy": "mean (numeric)", "fill_value": num_means[c]}
        for c in num_cols
    ] + [
        {"feature": c, "strategy": "mode (categorical)", "fill_value": cat_modes.get(c, "N/A")}
        for c in cat_cols
    ])

    # Step 3 – one-hot encoding
    X_imputed = X_unk.copy()
    for col in num_cols:
        X_imputed[col] = X_imputed[col].fillna(num_means[col])
    for col in cat_cols:
        X_imputed[col] = X_imputed[col].fillna(cat_modes.get(col, "unknown"))

    X_encoded = pd.get_dummies(X_imputed, drop_first=False)
    new_cols   = [c for c in X_encoded.columns if c not in num_cols]
    ohe_summary = pd.DataFrame({
        "original_feature": cat_cols,
        "n_dummy_columns": [
            sum(1 for c in X_encoded.columns if c.startswith(orig + "_"))
            for orig in cat_cols
        ],
    })

    # Step 4 – standardization (show before/after for numeric cols)
    means = X_encoded[num_cols].mean()
    stds  = X_encoded[num_cols].std().replace(0, 1)
    X_scaled = X_encoded.copy()
    X_scaled[num_cols] = (X_encoded[num_cols] - means) / stds

    scaling_df = pd.DataFrame({
        "feature":      num_cols,
        "train_mean":   means.values.round(4),
        "train_std":    stds.values.round(4),
        "scaled_mean":  X_scaled[num_cols].mean().values.round(6),
        "scaled_std":   X_scaled[num_cols].std().values.round(6),
    })

    return {
        "n_features_original":  X_train_raw.shape[1],
        "n_features_encoded":   X_encoded.shape[1],
        "n_train_samples":      X_train_raw.shape[0],
        "n_test_samples":       X_test_raw.shape[0],
        "numerical_cols":       num_cols,
        "categorical_cols":     cat_cols,
        "unknown_counts":       unknown_counts,
        "has_unknowns":         bool(unknown_counts),
        "imputation_df":        imputation_df,
        "ohe_summary_df":       ohe_summary,
        "encoded_columns":      X_encoded.columns.tolist(),
        "scaling_df":           scaling_df,
        "steps": [
            "1. Replace 'unknown' strings with NaN",
            "2. Impute: mean for numerical, mode for categorical",
            "3. One-Hot Encode all categorical columns (drop_first=False)",
            "4. Standardize all numerical columns: z = (x - mean) / std",
        ],
    }


# ------ 5.4  Train pipeline ---------------------------------

def train_pipeline(X_train_raw: pd.DataFrame,
                   X_test_raw: pd.DataFrame,
                   y_train_raw: pd.DataFrame,
                   y_test_raw: pd.DataFrame,
                   learning_rate: float = 0.01,
                   n_iterations: int = 1000,
                   model_path: str = "lr_model.pkl",
                   preprocessor_path: str = "preprocessor.pkl") -> dict:
    """Preprocess, train, evaluate, and save artifacts.

    Returns a dict containing the trained model, preprocessor, processed arrays,
    loss history, feature names, and default-threshold metrics.
    Populates the module-level _PIPELINE_STATE cache.
    """
    # Encode targets
    y_train = encode_target(y_train_raw)
    y_test  = encode_target(y_test_raw)

    # Preprocess (no-leakage: fit only on train)
    preprocessor = Preprocessor()
    with contextlib.redirect_stdout(io.StringIO()):
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test  = preprocessor.transform(X_test_raw)

    feature_names = preprocessor.get_feature_names()

    # Train
    model = LogisticRegressionScratch(learning_rate=learning_rate,
                                      n_iterations=n_iterations)
    model.fit(X_train, y_train)

    # Default-threshold evaluation
    y_pred_train = model.predict(X_train, threshold=0.5)
    y_pred_test  = model.predict(X_test,  threshold=0.5)
    train_metrics = evaluate(y_train, y_pred_train)
    test_metrics  = evaluate(y_test,  y_pred_test)

    # Build loss-history DataFrame for plotting
    loss_df = pd.DataFrame({
        "iteration": list(range(1, len(model.loss_history) + 1)),
        "loss":      model.loss_history,
    })

    # Save artifacts
    save_artifacts(model, preprocessor, model_path, preprocessor_path)

    result = {
        "model":           model,
        "preprocessor":    preprocessor,
        "X_train":         X_train,
        "X_test":          X_test,
        "y_train":         y_train,
        "y_test":          y_test,
        "X_train_raw":     X_train_raw,
        "X_test_raw":      X_test_raw,
        "feature_names":   feature_names,
        "n_features":      X_train.shape[1],
        "learning_rate":   learning_rate,
        "n_iterations":    n_iterations,
        "loss_history":    model.loss_history,
        "loss_df":         loss_df,
        "final_loss":      model.loss_history[-1],
        "train_metrics":   train_metrics,
        "test_metrics":    test_metrics,
        "model_path":      model_path,
        "preprocessor_path": preprocessor_path,
    }

    # Populate module-level cache so predict_single() works without re-training
    _PIPELINE_STATE.update(result)

    return result


# ------ 5.5  Evaluation -------------------------------------

def run_evaluation(model: LogisticRegressionScratch,
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray,  y_test: np.ndarray,
                   threshold: float = 0.5) -> dict:
    """Evaluate the trained model at a given threshold.

    Returns metrics, confusion matrices, and sample predictions as dicts /
    DataFrames for direct display in Streamlit.
    """
    y_pred_train = model.predict(X_train, threshold=threshold)
    y_pred_test  = model.predict(X_test,  threshold=threshold)

    train_metrics = evaluate(y_train, y_pred_train)
    test_metrics  = evaluate(y_test,  y_pred_test)

    tp_tr, fp_tr, tn_tr, fn_tr = confusion_matrix_values(y_train, y_pred_train)
    tp_te, fp_te, tn_te, fn_te = confusion_matrix_values(y_test,  y_pred_test)

    metrics_df = pd.DataFrame({
        "Metric":    ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Train":     [train_metrics[k] for k in ["Accuracy", "Precision", "Recall", "F1-Score"]],
        "Test":      [test_metrics[k]  for k in ["Accuracy", "Precision", "Recall", "F1-Score"]],
    }).round(4)

    # Sample predictions (first 15 test samples)
    y_proba = model.predict_proba(X_test)
    sample_df = pd.DataFrame({
        "#":           range(1, 16),
        "P(subscribe)": [round(float(p), 4) for p in y_proba[:15]],
        f"Pred@{threshold}": [int(p) for p in y_pred_test[:15]],
        "Actual":       [int(a) for a in y_test[:15]],
        "Correct":      [int(p) == int(a) for p, a in
                         zip(y_pred_test[:15], y_test[:15])],
    })

    return {
        "threshold":        threshold,
        "train_metrics":    train_metrics,
        "test_metrics":     test_metrics,
        "metrics_df":       metrics_df,
        "train_cm":         {"TP": tp_tr, "FP": fp_tr, "TN": tn_tr, "FN": fn_tr},
        "test_cm":          {"TP": tp_te, "FP": fp_te, "TN": tn_te, "FN": fn_te},
        "sample_predictions_df": sample_df,
        "n_test_positive":  int(np.sum(y_test == 1)),
        "n_test_negative":  int(np.sum(y_test == 0)),
    }


# ------ 5.6  Threshold analysis -----------------------------

def run_threshold_analysis(model: LogisticRegressionScratch,
                            X_test: np.ndarray,
                            y_test: np.ndarray) -> dict:
    """Full threshold sensitivity + business analysis.

    Returns DataFrames and summary dicts for Streamlit charts & tables.
    """
    # Fine-grained sensitivity (0.1 → 0.9, step 0.1)
    thresholds_fine = [round(t * 0.1, 1) for t in range(1, 10)]
    sensitivity_rows = []
    for thresh in thresholds_fine:
        y_pred = model.predict(X_test, threshold=thresh)
        m = evaluate(y_test, y_pred)
        sensitivity_rows.append({
            "Threshold": thresh,
            "Accuracy":  round(m["Accuracy"],  4),
            "Precision": round(m["Precision"], 4),
            "Recall":    round(m["Recall"],    4),
            "F1-Score":  round(m["F1-Score"],  4),
        })
    sensitivity_df = pd.DataFrame(sensitivity_rows)

    best_f1_row  = sensitivity_df.loc[sensitivity_df["F1-Score"].idxmax()]
    best_rec_row = sensitivity_df.loc[sensitivity_df["Recall"].idxmax()]

    # Comparison at selected thresholds
    comparison_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    comparison_rows = []
    for thresh in comparison_thresholds:
        y_pred = model.predict(X_test, threshold=thresh)
        m = evaluate(y_test, y_pred)
        comparison_rows.append({
            "Threshold": thresh,
            "Accuracy":  round(m["Accuracy"],  4),
            "Precision": round(m["Precision"], 4),
            "Recall":    round(m["Recall"],    4),
            "F1-Score":  round(m["F1-Score"],  4),
        })
    comparison_df = pd.DataFrame(comparison_rows)

    # Business analysis: 0.5 vs 0.3
    y_pred_05 = model.predict(X_test, threshold=0.5)
    y_pred_03 = model.predict(X_test, threshold=0.3)
    tp_05, fp_05, tn_05, fn_05 = confusion_matrix_values(y_test, y_pred_05)
    tp_03, fp_03, tn_03, fn_03 = confusion_matrix_values(y_test, y_pred_03)
    total_pos = int(np.sum(y_test == 1))

    business_df = pd.DataFrame([
        {"Threshold": 0.5, "TP": tp_05, "FP": fp_05, "TN": tn_05, "FN": fn_05,
         "Recall": round(tp_05 / total_pos, 4) if total_pos else 0},
        {"Threshold": 0.3, "TP": tp_03, "FP": fp_03, "TN": tn_03, "FN": fn_03,
         "Recall": round(tp_03 / total_pos, 4) if total_pos else 0},
    ])

    return {
        "sensitivity_df":       sensitivity_df,
        "comparison_df":        comparison_df,
        "best_f1_threshold":    float(best_f1_row["Threshold"]),
        "best_f1_value":        float(best_f1_row["F1-Score"]),
        "best_recall_threshold": float(best_rec_row["Threshold"]),
        "best_recall_value":    float(best_rec_row["Recall"]),
        "business_df":          business_df,
        "business_summary": {
            "total_actual_positives": total_pos,
            "threshold_05": {"TP": tp_05, "FP": fp_05, "FN": fn_05},
            "threshold_03": {"TP": tp_03, "FP": fp_03, "FN": fn_03},
            "extra_tp_from_lowering": tp_03 - tp_05,
            "extra_fp_from_lowering": fp_03 - fp_05,
        },
        "interpretation": (
            "Lowering the threshold increases Recall (fewer missed subscribers) "
            "but decreases Precision (more wasted calls). "
            f"Best F1 at threshold={float(best_f1_row['Threshold']):.1f} "
            f"(F1={float(best_f1_row['F1-Score']):.4f}). "
            "For a telemarketing campaign, threshold=0.2–0.3 is typically preferred "
            "because missing a subscriber costs more than a wasted call."
        ),
    }


# ------ 5.7  Feature analysis -------------------------------

def run_feature_analysis(model: LogisticRegressionScratch,
                          preprocessor: Preprocessor,
                          X_train_raw: pd.DataFrame,
                          X_test_raw: pd.DataFrame,
                          y_train: np.ndarray,
                          y_test: np.ndarray,
                          top_n: int = 10) -> dict:
    """Coefficient analysis + LR sensitivity + duration ablation.

    All experiments return DataFrames / dicts suitable for Streamlit charts.
    """
    feature_names = preprocessor.get_feature_names()

    # ── Coefficient DataFrame ────────────────────────────────
    weights = model.weights
    coef_df = pd.DataFrame({
        "feature":     feature_names,
        "coefficient": weights,
    }).sort_values("coefficient", ascending=False).reset_index(drop=True)
    coef_df["direction"] = coef_df["coefficient"].apply(
        lambda w: "positive" if w > 0 else "negative"
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()

    top_positive_df = coef_df.head(top_n).copy()
    top_negative_df = coef_df.tail(top_n).sort_values("coefficient").copy()

    # ── LR sensitivity (suppressed print; return only) ───────
    with contextlib.redirect_stdout(io.StringIO()):
        X_train = preprocessor.transform(X_train_raw)
        X_test  = preprocessor.transform(X_test_raw)

    lr_results = run_lr_sensitivity(X_train, y_train, X_test, y_test)
    lr_df = pd.DataFrame(lr_results).round(6)
    lr_df.columns = ["LR", "Iters", "Final Loss",
                     "Accuracy", "Precision", "Recall", "F1-Score"]

    # ── Duration ablation ────────────────────────────────────
    ablation = run_duration_ablation(X_train_raw, X_test_raw, y_train, y_test)
    ablation_df = pd.DataFrame([
        {
            "Setting":   "Full Features",
            "Accuracy":  round(ablation["full_features"]["Accuracy"],  4),
            "Precision": round(ablation["full_features"]["Precision"], 4),
            "Recall":    round(ablation["full_features"]["Recall"],    4),
            "F1-Score":  round(ablation["full_features"]["F1-Score"],  4),
        },
        {
            "Setting":   "Without Duration",
            "Accuracy":  round(ablation["without_duration"]["Accuracy"],  4),
            "Precision": round(ablation["without_duration"]["Precision"], 4),
            "Recall":    round(ablation["without_duration"]["Recall"],    4),
            "F1-Score":  round(ablation["without_duration"]["F1-Score"],  4),
        },
    ])

    return {
        "coef_df":           coef_df,
        "top_positive_df":   top_positive_df,
        "top_negative_df":   top_negative_df,
        "lr_sensitivity_df": lr_df,
        "lr_raw":            lr_results,
        "ablation_df":       ablation_df,
        "ablation_raw":      ablation,
        "top_n":             top_n,
        "most_positive_feature":  coef_df.iloc[0]["feature"],
        "most_positive_coef":     round(float(coef_df.iloc[0]["coefficient"]), 4),
        "most_negative_feature":  coef_df.iloc[-1]["feature"],
        "most_negative_coef":     round(float(coef_df.iloc[-1]["coefficient"]), 4),
    }


# ------ 5.8  Single-sample prediction -----------------------

def predict_single(input_dict: dict,
                   model: "LogisticRegressionScratch | None" = None,
                   preprocessor: "Preprocessor | None" = None,
                   thresholds: list | None = None) -> dict:
    """Predict subscription probability for one client.

    Parameters
    ----------
    input_dict : dict
        Raw feature values, e.g.::

            {
                "age": 35, "job": "management", "marital": "married",
                "education": "tertiary", "default": "no", "balance": 1500,
                "housing": "yes", "loan": "no", "contact": "cellular",
                "day": 15, "month": "may", "duration": 200,
                "campaign": 2, "pdays": -1, "previous": 0, "poutcome": "unknown"
            }

    model, preprocessor : optional
        If omitted, uses the module-level _PIPELINE_STATE cache (populated by
        train_pipeline() or run_full_demo()).

    thresholds : list of float, optional
        Decision thresholds to evaluate. Defaults to [0.1, 0.2, 0.3, 0.5].

    Returns
    -------
    dict with probability, per-threshold predictions, and interpretation.
    """
    if model is None:
        model = _PIPELINE_STATE.get("model")
    if preprocessor is None:
        preprocessor = _PIPELINE_STATE.get("preprocessor")

    if model is None or preprocessor is None:
        raise RuntimeError(
            "No trained model found. Call train_pipeline() or run_full_demo() first, "
            "or pass model and preprocessor explicitly."
        )

    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.5]

    # Build a complete single-row DataFrame. This lets a frontend expose only
    # the user-facing fields while the trained preprocessor fills the rest from
    # training-set statistics instead of producing NaNs.
    sample_dict = dict(input_dict)
    for col in getattr(preprocessor, "numerical_cols_", []) or []:
        if col not in sample_dict:
            sample_dict[col] = float(preprocessor.numerical_means_[col])
    for col in getattr(preprocessor, "categorical_cols_", []) or []:
        if col not in sample_dict:
            sample_dict[col] = preprocessor.categorical_modes_[col]

    sample_df = pd.DataFrame([sample_dict])

    # Transform (handles unknown→NaN, imputation, OHE, scaling)
    X_sample = preprocessor.transform(sample_df)
    if not np.isfinite(X_sample).all():
        X_sample = np.nan_to_num(X_sample, nan=0.0, posinf=0.0, neginf=0.0)

    prob = float(model.predict_proba(X_sample)[0])
    if not np.isfinite(prob):
        prob = 0.0

    predictions = {
        round(t, 2): int(prob >= t)
        for t in thresholds
    }

    # Human-readable risk level
    if prob >= 0.5:
        risk_level = "High"
        risk_color = "green"
        interpretation = (
            f"The model is confident this client will subscribe "
            f"(P={prob:.1%}). Prioritise for outreach."
        )
    elif prob >= 0.3:
        risk_level = "Medium"
        risk_color = "orange"
        interpretation = (
            f"Moderate subscription probability (P={prob:.1%}). "
            f"Worth contacting, especially with a lower-threshold campaign."
        )
    elif prob >= 0.15:
        risk_level = "Low-Medium"
        risk_color = "orange"
        interpretation = (
            f"Below-average probability (P={prob:.1%}). "
            f"Include only when campaign recall is prioritised over precision."
        )
    else:
        risk_level = "Low"
        risk_color = "red"
        interpretation = (
            f"Low subscription probability (P={prob:.1%}). "
            f"Unlikely to subscribe; deprioritise for this campaign."
        )

    predictions_df = pd.DataFrame([
        {"Threshold": t, "Predict Subscribe": bool(p), "Label": "Yes" if p else "No"}
        for t, p in predictions.items()
    ])

    return {
        "input":            input_dict,
        "probability":      round(prob, 6),
        "probability_pct":  f"{prob:.2%}",
        "predictions":      predictions,
        "predictions_df":   predictions_df,
        "risk_level":       risk_level,
        "risk_color":       risk_color,
        "interpretation":   interpretation,
    }


# ------ 5.9  Grid-search wrapper (frontend) -----------------

def run_grid_search(X_train_raw: pd.DataFrame,
                    X_test_raw: pd.DataFrame,
                    y_train: np.ndarray,
                    y_test: np.ndarray) -> dict:
    """Convenience wrapper: preprocess then call run_hyperparameter_grid_search.

    Returns the full grid-search result dict (see run_hyperparameter_grid_search).
    """
    prep = Preprocessor()
    with contextlib.redirect_stdout(io.StringIO()):
        X_tr = prep.fit_transform(X_train_raw)
        X_te = prep.transform(X_test_raw)

    result = run_hyperparameter_grid_search(
        X_tr, y_train, X_te, y_test,
        X_train_raw, X_test_raw,
    )

    result["all_results_df"] = pd.DataFrame(result["all_results"]).round(6)
    result["top10_df"]       = pd.DataFrame(result["top10"]).round(6)
    result["top10_df"].insert(0, "Rank", range(1, len(result["top10"]) + 1))

    return result


# ------ 5.10  Master demo function --------------------------

def run_full_demo(learning_rate: float = 0.01,
                  n_iterations: int = 1000,
                  test_size: float = 0.2,
                  random_seed: int = 42,
                  model_path: str = "lr_model.pkl",
                  preprocessor_path: str = "preprocessor.pkl") -> dict:
    """Run the complete ML workflow and return all results in one dict.

    This is the single entry point for a Streamlit frontend:

        import MODEL1_FINAL_前端 as pipeline
        results = pipeline.run_full_demo()
        st.write(results["dataset_summary"]["class_distribution_df"])
        ...

    Steps (in order):
        1. Load dataset
        2. Dataset exploration
        3. Feature descriptions
        4. Preprocessing demo
        5. Train/test split + training
        6. Default-threshold evaluation
        7. Threshold analysis
        8. Feature & coefficient analysis
        9. Sample prediction (demo client)
       10. Grid search (15 models × 8 thresholds)
    """
    print("\n" + "=" * 60)
    print("  BANK MARKETING — FULL DEMO PIPELINE")
    print("=" * 60 + "\n")

    # Step 1 – Load
    X_raw, y_raw = load_data()
    print_dataset_summary(X_raw, y_raw)
    print_unknown_summary(X_raw)

    # Step 2 – Dataset summary
    dataset_summary = get_dataset_summary(X_raw, y_raw)

    # Step 3 – Feature descriptions
    feature_descriptions = get_feature_descriptions()

    # Step 4 – Split (before preprocessing — no leakage)
    print("\nSplitting data (BEFORE preprocessing to prevent leakage)...")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split_df(
        X_raw, y_raw, test_size=test_size, random_seed=random_seed
    )
    print(f"  Train samples: {X_train_raw.shape[0]}")
    print(f"  Test samples:  {X_test_raw.shape[0]}")

    dataset_summary["train_samples"] = X_train_raw.shape[0]
    dataset_summary["test_samples"]  = X_test_raw.shape[0]
    dataset_summary["test_size"]     = test_size
    dataset_summary["random_seed"]   = random_seed

    # Step 4b – Preprocessing demo
    print("\nBuilding preprocessing demo...")
    preprocessing_demo = run_preprocessing_demo(X_train_raw, X_test_raw)

    # Step 5 – Train
    print("\nTraining Logistic Regression...")
    pipeline_result = train_pipeline(
        X_train_raw, X_test_raw, y_train_raw, y_test_raw,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        model_path=model_path,
        preprocessor_path=preprocessor_path,
    )
    model       = pipeline_result["model"]
    preprocessor = pipeline_result["preprocessor"]
    X_train     = pipeline_result["X_train"]
    X_test      = pipeline_result["X_test"]
    y_train     = pipeline_result["y_train"]
    y_test      = pipeline_result["y_test"]

    feature_names = pipeline_result["feature_names"]
    print_top_coefficients(model, feature_names, top_n=10)

    # Step 6 – Evaluation (default 0.5)
    print("\n" + "=" * 60)
    print("  EVALUATION WITH DEFAULT THRESHOLD (0.5)")
    print("=" * 60)
    evaluation_05 = run_evaluation(model, X_train, y_train, X_test, y_test, threshold=0.5)
    print_results(evaluation_05["train_metrics"], split_name="Train")
    print_confusion_matrix(y_train,
                           model.predict(X_train, threshold=0.5),
                           title="Confusion Matrix (Train Set)")
    print_results(evaluation_05["test_metrics"], split_name="Test")
    print_confusion_matrix(y_test,
                           model.predict(X_test, threshold=0.5),
                           title="Confusion Matrix (Test Set)")

    # Evaluation at 0.3 (detailed)
    print("\n" + "=" * 60)
    print("  DETAILED EVALUATION WITH THRESHOLD = 0.3")
    print("=" * 60)
    evaluation_03 = run_evaluation(model, X_train, y_train, X_test, y_test, threshold=0.3)
    print_results(evaluation_03["test_metrics"], split_name="Test (threshold=0.3)")
    print_confusion_matrix(y_test,
                           model.predict(X_test, threshold=0.3),
                           title="Confusion Matrix (Test, threshold=0.3)")

    # Step 7 – Threshold analysis
    print("\n" + "=" * 60)
    print("  THRESHOLD TUNING ANALYSIS")
    print("=" * 60)
    print_threshold_comparison(model, X_test, y_test,
                                [0.3, 0.4, 0.5, 0.6, 0.7], split_name="Test")
    print_threshold_sensitivity(model, X_test, y_test)
    print_business_threshold_analysis(model, X_test, y_test)
    threshold_analysis = run_threshold_analysis(model, X_test, y_test)

    # Step 8 – Feature analysis (includes LR sensitivity + ablation)
    print("\n" + "=" * 60)
    print("  EXPERIMENT: LEARNING RATE / ITERATION SENSITIVITY")
    print("=" * 60)
    print("\n" + "=" * 60)
    print("  EXPERIMENT: FEATURE SENSITIVITY (DURATION ABLATION)")
    print("=" * 60)
    feature_analysis = run_feature_analysis(
        model, preprocessor,
        X_train_raw, X_test_raw,
        y_train, y_test,
        top_n=10,
    )

    # Step 9 – Sample prediction (demo client)
    demo_client = {
        "age":       35,
        "job":       "management",
        "marital":   "married",
        "education": "tertiary",
        "default":   "no",
        "balance":   1500,
        "housing":   "yes",
        "loan":      "no",
        "contact":   "cellular",
        "day":       15,
        "month":     "may",
        "duration":  200,
        "campaign":  2,
        "pdays":     -1,
        "previous":  0,
        "poutcome":  "failure",
    }
    sample_prediction = predict_single(demo_client, model, preprocessor)

    print("\n" + "=" * 60)
    print("  SAMPLE PREDICTION (demo client)")
    print("=" * 60)
    print(f"  Input    : {demo_client}")
    print(f"  P(subscribe) = {sample_prediction['probability_pct']}")
    print(f"  Risk level   : {sample_prediction['risk_level']}")
    print(f"  Interpretation: {sample_prediction['interpretation']}")

    # Step 10 – Grid search
    print("\n" + "=" * 60)
    print("  EXPERIMENT: HYPERPARAMETER GRID SEARCH")
    print("=" * 60)
    grid_search = run_grid_search(X_train_raw, X_test_raw, y_train, y_test)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60 + "\n")

    return {
        # ── Raw data (for advanced use) ──────────────────────
        "X_raw":            X_raw,
        "y_raw":            y_raw,
        "X_train_raw":      X_train_raw,
        "X_test_raw":       X_test_raw,
        # ── Trained artefacts ────────────────────────────────
        "model":            model,
        "preprocessor":     preprocessor,
        "X_train":          X_train,
        "X_test":           X_test,
        "y_train":          y_train,
        "y_test":           y_test,
        # ── Section results (frontend-ready) ─────────────────
        "dataset_summary":      dataset_summary,
        "feature_descriptions": feature_descriptions,
        "preprocessing_demo":   preprocessing_demo,
        "pipeline_result":      pipeline_result,
        "evaluation_05":        evaluation_05,
        "evaluation_03":        evaluation_03,
        "threshold_analysis":   threshold_analysis,
        "feature_analysis":     feature_analysis,
        "sample_prediction":    sample_prediction,
        "grid_search":          grid_search,
    }


# ============================================================
# SECTION 6 – ORIGINAL CLI ENTRY POINT  (unchanged)
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  BANK MARKETING LOGISTIC REGRESSION PIPELINE")
    print("  UCI Bank Marketing Dataset (id=222)")
    print("  (No Data Leakage - Split Before Preprocessing)")
    print("=" * 60 + "\n")

    X_raw, y_raw = load_data()
    print_dataset_summary(X_raw, y_raw)

    print_unknown_summary(X_raw)

    print("\nSplitting data (BEFORE preprocessing to prevent leakage)...")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split_df(
        X_raw, y_raw, test_size=0.2, random_seed=42
    )
    print(f"  Train samples: {X_train_raw.shape[0]}")
    print(f"  Test samples:  {X_test_raw.shape[0]}")

    y_train = encode_target(y_train_raw)
    y_test = encode_target(y_test_raw)

    print("\nPreprocessing:")
    preprocessor = Preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)

    print("  Transforming test data using training parameters...")
    X_test = preprocessor.transform(X_test_raw)

    print(f"\n  Final feature count: {X_train.shape[1]}")

    print("\nTraining Logistic Regression...")
    model = LogisticRegressionScratch(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)

    feature_names = preprocessor.get_feature_names()
    print_top_coefficients(model, feature_names, top_n=10)

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

    print("\n" + "=" * 60)
    print("  THRESHOLD TUNING ANALYSIS")
    print("=" * 60)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print_threshold_comparison(model, X_test, y_test, thresholds, split_name="Test")

    print_threshold_sensitivity(model, X_test, y_test)

    print_business_threshold_analysis(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("  DETAILED EVALUATION WITH THRESHOLD = 0.3")
    print("  (Lower threshold to improve recall for imbalanced data)")
    print("=" * 60)

    y_pred_test_03 = model.predict(X_test, threshold=0.3)
    test_metrics_03 = evaluate(y_test, y_pred_test_03)

    print_results(test_metrics_03, split_name="Test (threshold=0.3)")
    print_confusion_matrix(y_test, y_pred_test_03, title="Confusion Matrix (Test, threshold=0.3)")

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
    print("  EXPERIMENT: LEARNING RATE / ITERATION SENSITIVITY")
    print("=" * 60)
    run_lr_sensitivity(X_train, y_train, X_test, y_test)

    print("\n" + "=" * 60)
    print("  EXPERIMENT: FEATURE SENSITIVITY (DURATION ABLATION)")
    print("=" * 60)
    run_duration_ablation(X_train_raw, X_test_raw, y_train, y_test)

    print("\n" + "=" * 60)
    print("  EXPERIMENT: HYPERPARAMETER GRID SEARCH")
    print("=" * 60)
    run_hyperparameter_grid_search(
        X_train, y_train, X_test, y_test,
        X_train_raw, X_test_raw
    )

    print("\n" + "=" * 60)
    print("  SAVING ARTIFACTS")
    print("=" * 60)
    save_artifacts(model, preprocessor,
                   model_path="lr_model.pkl",
                   preprocessor_path="preprocessor.pkl")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
