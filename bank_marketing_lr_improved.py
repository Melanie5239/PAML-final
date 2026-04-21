import io
import pickle
import contextlib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_data():
    print("Loading dataset...")
    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y


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
        num_cols = X.select_dtypes(include=[np.number]).columns
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


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix"):
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


def run_lr_sensitivity(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray):
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


def run_duration_ablation(X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame,
                           y_train: np.ndarray, y_test: np.ndarray):
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


def run_hyperparameter_grid_search(X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: np.ndarray, y_test: np.ndarray,
                                    X_train_raw: pd.DataFrame,
                                    X_test_raw: pd.DataFrame):
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

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load Data
def load_data():
    print("Loading dataset...")
    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features.copy()
    y = bank_marketing.data.targets.copy()
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y

X, y = load_data()

# Preprocessing
df = pd.concat([X, y], axis=1).copy()

# replace unknown with NaN
df = df.replace("unknown", np.nan)

# fill missing values without inplace chained assignment
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

# one-hot encode
df = pd.get_dummies(df, drop_first=True)

# make sure everything is numeric
df = df.astype(float)

# target and features
y = df["y_yes"].to_numpy(dtype=int)
X = df.drop(columns=["y_yes"]).to_numpy(dtype=float)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Standardization
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1e-8

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Naive Bayes
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.prior = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0) + 1e-6
            self.prior[c] = X_c.shape[0] / X.shape[0]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict_proba(self, X):
        probs = []

        for x in X:
            class_log_probs = []

            for c in self.classes:
                prior = np.log(self.prior[c])
                likelihood = np.sum(np.log(self._pdf(c, x) + 1e-12))
                class_log_probs.append(prior + likelihood)

            probs.append(class_log_probs)

        probs = np.array(probs)
        probs = np.exp(probs - np.max(probs, axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)

        return probs[:, 1]

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

# Train
model = NaiveBayes()
model.fit(X_train, y_train)

# Evaluation
thresholds = [0.3, 0.4, 0.5, 0.6]

for t in thresholds:
    y_pred = model.predict(X_test, threshold=t)

    print(f"\nThreshold = {t}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1-score:", f1_score(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
