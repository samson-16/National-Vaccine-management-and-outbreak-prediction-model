"""Train the measles MVP next-month outbreak risk model.

This script is designed for the final-year project demo:

- Ethiopia woreda-month rows are the only rows used for Ethiopia evaluation.
- Public external-country measles rows are summarized into safe outbreak-pattern
  priors, then merged by month.
- The model predicts next-month outbreak risk, not the same-month label.
- Leakage columns such as current target cases and current target outbreak are
  excluded from the feature matrix.
- XGBoost is treated as the primary model for the project, while Random Forest
  and KNN are retained as comparison models for the academic defense.

The script prefers scikit-learn when it is installed. If it is unavailable, it
falls back to small NumPy/Pandas implementations of logistic regression and a
random-forest-like ensemble so the demo remains runnable offline.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ETHIOPIA_INPUT = ROOT / "data" / "processed" / "measles_training_real_model_matrix.csv"
DEFAULT_EXTERNAL_INPUT_NEW = ROOT / "data" / "processed" / "external_measles_normalized_new.csv"
DEFAULT_EXTERNAL_INPUT = ROOT / "data" / "processed" / "external_measles_normalized.csv"
DEFAULT_OUTPUT_DIR = ROOT / "model_outputs_xgboost"
DEFAULT_REPORT = ROOT / "reports" / "training_readiness_report.md"
RANDOM_STATE = 42

LEAKAGE_COLUMNS = {
    "target_cases",
    "target_deaths",
    "target_outbreak",
    "target_outbreak_next_month",
    "target_cases_next_month",
    "target_deaths_next_month",
}

META_COLUMNS = {
    "location_id",
    "period_start",
    "prediction_period_start",
    "admin3_pcode",
}

PROVENANCE_OR_FLAG_COLUMNS = {
    "label_type",
    "real_flag",
    "weak_label_flag",
    "synthetic_flag",
    "imputed_flag",
    "guessed_flag",
    "external_country_flag",
    "ethiopia_flag",
    "source_notes",
    "generation_context",
}

CATEGORICAL_COLUMNS = [
    "admin1_region",
    "admin2_zone",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_ETHIOPIA_INPUT)
    parser.add_argument("--external", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--horizon-months", type=int, default=1)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--no-external-priors", action="store_true")
    parser.add_argument("--primary-model", type=str, default="xgboost")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def resolve_external_path(path: Path | None) -> Path | None:
    if path and path.exists():
        return path
    if DEFAULT_EXTERNAL_INPUT_NEW.exists():
        return DEFAULT_EXTERNAL_INPUT_NEW
    if DEFAULT_EXTERNAL_INPUT.exists():
        return DEFAULT_EXTERNAL_INPUT
    return None


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def weighted_positive_rate(y: np.ndarray, w: np.ndarray) -> float:
    total = float(np.sum(w))
    if total <= 0:
        return float(np.mean(y)) if len(y) else 0.0
    return float(np.sum(w * y) / total)


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    n = len(y)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    weights = np.ones(n, dtype=float)
    if pos > 0:
        weights[y == 1] = n / (2.0 * pos)
    if neg > 0:
        weights[y == 0] = n / (2.0 * neg)
    return weights


def auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    pos_rank_sum = float(np.sum(ranks[y_true == 1]))
    auc = (pos_rank_sum - (pos * (pos + 1) / 2.0)) / (pos * neg)
    return float(auc)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos = int(np.sum(y_true == 1))
    if pos == 0:
        return None
    order = np.argsort(-scores)
    sorted_y = y_true[order]
    tp = np.cumsum(sorted_y == 1)
    fp = np.cumsum(sorted_y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = (probabilities >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    beta2 = 4.0
    f2 = ((1 + beta2) * precision * recall / ((beta2 * precision) + recall)) if ((beta2 * precision) + recall) else 0.0
    return {
        "rows": int(len(y_true)),
        "positive_rows": int(np.sum(y_true == 1)),
        "negative_rows": int(np.sum(y_true == 0)),
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y_true)) if len(y_true) else None,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "f2": float(f2),
        "roc_auc": auc_rank(y_true, probabilities),
        "pr_auc": average_precision(y_true, probabilities),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def tune_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        score = compute_metrics(y_true, probabilities, float(threshold))["f2"]
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


class FeaturePreprocessor:
    def __init__(self) -> None:
        self.feature_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.medians: pd.Series | None = None

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        matrix = self._build_matrix(df, fit=True)
        self.feature_columns = list(matrix.columns)
        return matrix.to_numpy(dtype=float)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        matrix = self._build_matrix(df, fit=False)
        matrix = matrix.reindex(columns=self.feature_columns, fill_value=0.0)
        return matrix.to_numpy(dtype=float)

    def to_state(self) -> dict[str, Any]:
        if self.medians is None:
            raise RuntimeError("FeaturePreprocessor must be fit before export.")
        return {
            "feature_columns": self.feature_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "medians": {str(key): float(value) for key, value in self.medians.to_dict().items()},
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "FeaturePreprocessor":
        preprocessor = cls()
        preprocessor.feature_columns = list(state["feature_columns"])
        preprocessor.numeric_columns = list(state["numeric_columns"])
        preprocessor.categorical_columns = list(state["categorical_columns"])
        preprocessor.medians = pd.Series(state["medians"], dtype=float)
        return preprocessor

    def _build_matrix(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        excluded = LEAKAGE_COLUMNS | META_COLUMNS | PROVENANCE_OR_FLAG_COLUMNS
        candidate_columns = [c for c in df.columns if c not in excluded]
        categorical = [c for c in CATEGORICAL_COLUMNS if c in candidate_columns]
        numeric = [
            c
            for c in candidate_columns
            if c not in categorical and pd.api.types.is_numeric_dtype(df[c])
        ]
        if fit:
            self.numeric_columns = numeric
            self.categorical_columns = categorical
            self.medians = df[numeric].apply(pd.to_numeric, errors="coerce").median().fillna(0.0)
        if self.medians is None:
            raise RuntimeError("FeaturePreprocessor must be fit before transform.")
        numeric_frame = df[self.numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(self.medians)
        categorical_frame = df[self.categorical_columns].fillna("missing").astype(str)
        if len(self.categorical_columns):
            dummies = pd.get_dummies(categorical_frame, prefix=self.categorical_columns, dtype=float)
            return pd.concat([numeric_frame.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        return numeric_frame.reset_index(drop=True)


class NumpyLogisticRegression:
    def __init__(self, learning_rate: float = 0.05, n_iter: int = 2500, l2: float = 0.01) -> None:
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.l2 = l2
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyLogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Xs = (X - self.mean_) / self.std_
        Xb = np.c_[np.ones(len(Xs)), Xs]
        weights = class_balanced_weights(y.astype(int))
        coef = np.zeros(Xb.shape[1], dtype=float)
        weight_sum = float(np.sum(weights))
        for _ in range(self.n_iter):
            probabilities = sigmoid(Xb @ coef)
            gradient = (Xb.T @ ((probabilities - y) * weights)) / weight_sum
            gradient[1:] += self.l2 * coef[1:]
            coef -= self.learning_rate * gradient
        self.coef_ = coef
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None or self.coef_ is None:
            raise RuntimeError("Model must be fit before predict_proba.")
        Xs = (np.asarray(X, dtype=float) - self.mean_) / self.std_
        Xb = np.c_[np.ones(len(Xs)), Xs]
        return sigmoid(Xb @ self.coef_)

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        if self.coef_ is None:
            raise RuntimeError("Model must be fit before feature_importance.")
        importance = np.abs(self.coef_[1:])
        return pd.DataFrame({"feature": feature_names, "importance": importance})


@dataclass
class TreeNode:
    probability: float
    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


class DecisionTreeLite:
    def __init__(
        self,
        max_depth: int = 7,
        min_samples_leaf: int = 10,
        max_features: int | None = None,
        n_thresholds: int = 7,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.random_state = random_state
        self.root_: TreeNode | None = None
        self.feature_importances_: np.ndarray | None = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "DecisionTreeLite":
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=int)
        self.w_ = np.asarray(sample_weight if sample_weight is not None else np.ones(len(y)), dtype=float)
        self.feature_importances_ = np.zeros(self.X_.shape[1], dtype=float)
        indices = np.arange(len(self.y_))
        self.root_ = self._build(indices, depth=0)
        total = float(np.sum(self.feature_importances_))
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _gini(self, indices: np.ndarray) -> float:
        if len(indices) == 0:
            return 0.0
        rate = weighted_positive_rate(self.y_[indices], self.w_[indices])
        return float(1.0 - (rate * rate) - ((1.0 - rate) * (1.0 - rate)))

    def _build(self, indices: np.ndarray, depth: int) -> TreeNode:
        probability = weighted_positive_rate(self.y_[indices], self.w_[indices])
        node = TreeNode(probability=probability)
        if (
            depth >= self.max_depth
            or len(indices) < self.min_samples_leaf * 2
            or len(np.unique(self.y_[indices])) == 1
        ):
            return node

        n_features = self.X_.shape[1]
        max_features = self.max_features or max(1, int(math.sqrt(n_features)))
        feature_indices = self._rng.choice(n_features, size=min(max_features, n_features), replace=False)
        parent_gini = self._gini(indices)
        parent_weight = float(np.sum(self.w_[indices]))
        best_gain = 0.0
        best_feature: int | None = None
        best_threshold: float | None = None
        best_left: np.ndarray | None = None
        best_right: np.ndarray | None = None

        for feature_index in feature_indices:
            values = self.X_[indices, feature_index]
            if np.all(values == values[0]):
                continue
            quantiles = np.linspace(0.1, 0.9, self.n_thresholds)
            thresholds = np.unique(np.quantile(values, quantiles))
            for threshold in thresholds:
                left_mask = values <= threshold
                left_indices = indices[left_mask]
                right_indices = indices[~left_mask]
                if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                    continue
                left_weight = float(np.sum(self.w_[left_indices]))
                right_weight = float(np.sum(self.w_[right_indices]))
                child_gini = ((left_weight * self._gini(left_indices)) + (right_weight * self._gini(right_indices))) / parent_weight
                gain = parent_gini - child_gini
                if gain > best_gain:
                    best_gain = float(gain)
                    best_feature = int(feature_index)
                    best_threshold = float(threshold)
                    best_left = left_indices
                    best_right = right_indices

        if best_feature is None or best_threshold is None or best_left is None or best_right is None:
            return node

        if self.feature_importances_ is not None:
            self.feature_importances_[best_feature] += best_gain * parent_weight
        node.feature_index = best_feature
        node.threshold = best_threshold
        node.left = self._build(best_left, depth + 1)
        node.right = self._build(best_right, depth + 1)
        return node

    def _predict_row(self, row: np.ndarray, node: TreeNode) -> float:
        while node.feature_index is not None and node.threshold is not None and node.left is not None and node.right is not None:
            node = node.left if row[node.feature_index] <= node.threshold else node.right
        return float(node.probability)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("Tree must be fit before predict_proba.")
        return np.array([self._predict_row(row, self.root_) for row in np.asarray(X, dtype=float)])


class SimpleRandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 80,
        max_depth: int = 7,
        min_samples_leaf: int = 10,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees_: list[DecisionTreeLite] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleRandomForestClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        rng = np.random.default_rng(self.random_state)
        n_rows, n_features = X.shape
        max_features = max(1, int(math.sqrt(n_features)))
        weights = class_balanced_weights(y)
        self.trees_ = []
        importances = np.zeros(n_features, dtype=float)
        for tree_idx in range(self.n_estimators):
            bootstrap_indices = rng.choice(n_rows, size=n_rows, replace=True)
            tree = DecisionTreeLite(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=self.random_state + tree_idx + 1,
            )
            tree.fit(X[bootstrap_indices], y[bootstrap_indices], sample_weight=weights[bootstrap_indices])
            self.trees_.append(tree)
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_
        total = float(np.sum(importances))
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("Forest must be fit before predict_proba.")
        predictions = np.vstack([tree.predict_proba(X) for tree in self.trees_])
        return predictions.mean(axis=0)

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        if self.feature_importances_ is None:
            raise RuntimeError("Model must be fit before feature_importance.")
        return pd.DataFrame({"feature": feature_names, "importance": self.feature_importances_})


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


def make_sklearn_builders(random_state: int) -> dict[str, Callable[[], Any]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    class LogisticWrapper:
        def __init__(self) -> None:
            self.scaler = StandardScaler()
            self.model = LogisticRegression(max_iter=2500, class_weight="balanced", solver="liblinear", random_state=random_state)

        def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticWrapper":
            Xs = self.scaler.fit_transform(X)
            self.model.fit(Xs, y)
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return self.model.predict_proba(self.scaler.transform(X))[:, 1]

        def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
            return pd.DataFrame({"feature": feature_names, "importance": np.abs(self.model.coef_[0])})

    class RandomForestWrapper:
        def __init__(self) -> None:
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=1,
            )

        def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestWrapper":
            self.model.fit(X, y)
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return self.model.predict_proba(X)[:, 1]

        def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
            return pd.DataFrame({"feature": feature_names, "importance": self.model.feature_importances_})

    class KNNWrapper:
        def __init__(self) -> None:
            self.scaler = StandardScaler()
            self.model = KNeighborsClassifier(
                n_neighbors=15,
                weights="distance",
                metric="minkowski",
                n_jobs=1,
            )
            self.random_state = random_state
            self.max_train_rows = 35000
            self.negative_to_positive_ratio = 8

        def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNWrapper":
            X = np.asarray(X, dtype=float)
            y = np.asarray(y).astype(int)
            X, y = self._balanced_sample(X, y)
            self.scaler.fit(X)
            self.model.fit(self.scaler.transform(X), y)
            return self

        def _balanced_sample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            positive_idx = np.flatnonzero(y == 1)
            negative_idx = np.flatnonzero(y == 0)
            if len(positive_idx) == 0 or len(X) <= self.max_train_rows:
                return X, y

            rng = np.random.default_rng(self.random_state)
            negative_cap = max(0, self.max_train_rows - len(positive_idx))
            max_negative = min(
                len(negative_idx),
                len(positive_idx) * self.negative_to_positive_ratio,
                negative_cap,
            )
            sampled_negative = rng.choice(negative_idx, size=max_negative, replace=False)
            sampled = np.concatenate([positive_idx, sampled_negative])
            rng.shuffle(sampled)
            return X[sampled], y[sampled]

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return self.model.predict_proba(self.scaler.transform(X))[:, 1]

        def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
            return pd.DataFrame({"feature": feature_names, "importance": np.zeros(len(feature_names), dtype=float)})

    builders: dict[str, Callable[[], Any]] = {
        "logistic_regression": LogisticWrapper,
        "random_forest": RandomForestWrapper,
        "knn": KNNWrapper,
    }
    try:
        from xgboost import XGBClassifier

        class XGBoostWrapper:
            def __init__(self) -> None:
                self.model: XGBClassifier | None = None

            def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostWrapper":
                y = np.asarray(y).astype(int)
                positive = int(np.sum(y == 1))
                negative = int(np.sum(y == 0))
                scale_pos_weight = negative / positive if positive else 1.0
                self.model = XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    n_jobs=1,
                )
                self.model.fit(X, y)
                return self

            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                if self.model is None:
                    raise RuntimeError("Model must be fit before predict_proba.")
                return self.model.predict_proba(X)[:, 1]

            def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
                if self.model is None:
                    raise RuntimeError("Model must be fit before feature_importance.")
                return pd.DataFrame({"feature": feature_names, "importance": self.model.feature_importances_})

        builders["xgboost"] = XGBoostWrapper
    except Exception:
        pass
    return builders


def make_fallback_builders(random_state: int) -> dict[str, Callable[[], Any]]:
    return {
        "logistic_regression": lambda: NumpyLogisticRegression(),
        "random_forest_lite": lambda: SimpleRandomForestClassifier(random_state=random_state),
    }


@dataclass
class CandidateResult:
    name: str
    threshold: float
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    model: Any
    preprocessor: FeaturePreprocessor
    feature_columns: list[str]


def fit_candidate(
    name: str,
    builder: Callable[[], Any],
    threshold_fit_df: pd.DataFrame,
    threshold_valid_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> CandidateResult:
    threshold_preprocessor = FeaturePreprocessor()
    X_fit = threshold_preprocessor.fit_transform(threshold_fit_df)
    y_fit = threshold_fit_df["target_outbreak_next_month"].astype(int).to_numpy()
    X_valid = threshold_preprocessor.transform(threshold_valid_df)
    y_valid = threshold_valid_df["target_outbreak_next_month"].astype(int).to_numpy()
    threshold_model = builder()
    threshold_model.fit(X_fit, y_fit)
    valid_probabilities = threshold_model.predict_proba(X_valid)
    threshold = tune_threshold(y_valid, valid_probabilities)
    validation_metrics = compute_metrics(y_valid, valid_probabilities, threshold)

    final_preprocessor = FeaturePreprocessor()
    X_train = final_preprocessor.fit_transform(train_df)
    y_train = train_df["target_outbreak_next_month"].astype(int).to_numpy()
    final_model = builder()
    final_model.fit(X_train, y_train)
    X_test = final_preprocessor.transform(test_df)
    y_test = test_df["target_outbreak_next_month"].astype(int).to_numpy()
    test_probabilities = final_model.predict_proba(X_test)
    test_metrics = compute_metrics(y_test, test_probabilities, threshold)

    return CandidateResult(
        name=name,
        threshold=threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        model=final_model,
        preprocessor=final_preprocessor,
        feature_columns=final_preprocessor.feature_columns,
    )


def stratified_validation(
    df: pd.DataFrame,
    builder: Callable[[], Any],
    random_state: int,
    test_size: float = 0.2,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_state)
    y = df["target_outbreak_next_month"].astype(int).to_numpy()
    test_indices: list[int] = []
    for label in [0, 1]:
        label_indices = np.flatnonzero(y == label)
        rng.shuffle(label_indices)
        take = max(1, int(round(len(label_indices) * test_size))) if len(label_indices) else 0
        test_indices.extend(label_indices[:take].tolist())
    test_indices = sorted(set(test_indices))
    train_indices = sorted(set(range(len(df))) - set(test_indices))
    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()
    preprocessor = FeaturePreprocessor()
    X_train = preprocessor.fit_transform(train_df)
    y_train = train_df["target_outbreak_next_month"].astype(int).to_numpy()
    X_test = preprocessor.transform(test_df)
    y_test = test_df["target_outbreak_next_month"].astype(int).to_numpy()
    model = builder()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    threshold = tune_threshold(y_train, model.predict_proba(X_train))
    return compute_metrics(y_test, probabilities, threshold)


COUNT_COLUMNS = {
    "suspected_records",
    "target_cases",
    "target_deaths",
    "lab_confirmed_cases",
    "epi_linked_cases",
    "compatible_cases",
    "discarded_records",
    "other_final_classification_records",
    "case_based_records",
    "line_list_records",
    "under5_confirmed_cases",
    "under15_confirmed_cases",
    "zero_dose_confirmed_cases",
    "vaccinated_confirmed_cases",
    "unknown_vaccine_confirmed_cases",
}

LAG_COLUMNS = {
    "target_cases_lag_1",
    "target_cases_lag_2",
    "target_cases_lag_3",
    "target_cases_rolling_3_prev",
    "target_outbreak_lag_1",
}


def collapse_duplicate_location_months(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_count = int(df.duplicated(["location_id", "period_start"], keep=False).sum())
    if duplicate_count == 0:
        return df

    df = df.drop(columns=[column for column in LAG_COLUMNS if column in df.columns])
    group_columns = ["location_id", "period_start"]
    aggregations: dict[str, str] = {}
    for column in df.columns:
        if column in group_columns:
            continue
        if column in COUNT_COLUMNS:
            aggregations[column] = "sum"
        elif column == "target_outbreak":
            aggregations[column] = "max"
        elif pd.api.types.is_numeric_dtype(df[column]):
            aggregations[column] = "mean"
        else:
            aggregations[column] = "first"

    collapsed = df.groupby(group_columns, as_index=False).agg(aggregations)
    if "target_cases" in collapsed.columns:
        collapsed["target_outbreak"] = (pd.to_numeric(collapsed["target_cases"], errors="coerce").fillna(0) >= 5).astype(int)
    return collapsed


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["location_id", "period_start"]).reset_index(drop=True)
    grouped = df.groupby("location_id", group_keys=False)
    df["target_cases_lag_1"] = grouped["target_cases"].shift(1)
    df["target_cases_lag_2"] = grouped["target_cases"].shift(2)
    df["target_cases_lag_3"] = grouped["target_cases"].shift(3)
    df["target_cases_rolling_3_prev"] = grouped["target_cases"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["target_outbreak_lag_1"] = grouped["target_outbreak"].shift(1)
    for column in LAG_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return df


def load_ethiopia_matrix(path: Path, horizon_months: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Ethiopia training matrix: {path}")
    df = pd.read_csv(path)
    df["period_start"] = pd.to_datetime(df["period_start"])
    df = collapse_duplicate_location_months(df)
    df = add_lag_features(df)
    grouped = df.groupby("location_id", group_keys=False)
    df["target_outbreak_next_month"] = grouped["target_outbreak"].shift(-horizon_months)
    df["target_cases_next_month"] = grouped["target_cases"].shift(-horizon_months)
    if "target_deaths" in df.columns:
        df["target_deaths_next_month"] = grouped["target_deaths"].shift(-horizon_months)
    df["prediction_period_start"] = df["period_start"] + pd.DateOffset(months=horizon_months)
    return df


def default_month_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": list(range(1, 13)),
            "external_monthly_measles_seasonality_index": [0.5] * 12,
            "external_monthly_positive_rate": [0.0] * 12,
            "external_monthly_mean_log_cases": [0.0] * 12,
            "external_monthly_p90_log_cases": [0.0] * 12,
        }
    )


def compute_run_lengths(flags: list[bool]) -> list[int]:
    runs: list[int] = []
    current = 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def build_external_patterns(external_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if external_path is None or not external_path.exists():
        month_features = default_month_features()
        patterns = pd.DataFrame(
            [
                {
                    "pattern_scope": "global",
                    "month": np.nan,
                    "metric": "external_rows_available",
                    "value": 0,
                    "n_rows": 0,
                    "notes": "No external file found; neutral priors used.",
                }
            ]
        )
        return month_features, patterns, {"external_path": None, "external_rows": 0}

    ext = pd.read_csv(external_path)
    ext["cases"] = pd.to_numeric(ext.get("cases"), errors="coerce").fillna(0.0).clip(lower=0)
    ext["date_for_month"] = pd.to_datetime(ext.get("period_start", ext.get("date")), errors="coerce")
    ext = ext.dropna(subset=["date_for_month"]).copy()
    monthly = ext[ext.get("period_frequency", "").astype(str).str.lower().eq("monthly")].copy()
    if monthly.empty:
        monthly = ext.copy()
    monthly["month"] = monthly["date_for_month"].dt.month
    monthly["positive"] = monthly["cases"] > 0
    monthly["log_cases"] = np.log1p(monthly["cases"])

    by_month = (
        monthly.groupby("month")
        .agg(
            external_rows=("cases", "size"),
            external_locations=("location_id", "nunique"),
            external_total_cases=("cases", "sum"),
            external_positive_rate=("positive", "mean"),
            external_mean_log_cases=("log_cases", "mean"),
            external_p90_log_cases=("log_cases", lambda s: float(np.quantile(s, 0.9))),
        )
        .reset_index()
    )
    all_months = pd.DataFrame({"month": list(range(1, 13))})
    by_month = all_months.merge(by_month, on="month", how="left").fillna(0.0)
    raw = by_month["external_mean_log_cases"].to_numpy(dtype=float)
    if raw.max() > raw.min():
        seasonality = (raw - raw.min()) / (raw.max() - raw.min())
    elif by_month["external_positive_rate"].max() > 0:
        seasonality = by_month["external_positive_rate"] / by_month["external_positive_rate"].max()
    else:
        seasonality = np.full(len(by_month), 0.5)
    month_features = pd.DataFrame(
        {
            "month": by_month["month"].astype(int),
            "external_monthly_measles_seasonality_index": np.asarray(seasonality, dtype=float),
            "external_monthly_positive_rate": by_month["external_positive_rate"].astype(float),
            "external_monthly_mean_log_cases": by_month["external_mean_log_cases"].astype(float),
            "external_monthly_p90_log_cases": by_month["external_p90_log_cases"].astype(float),
        }
    )

    monthly = monthly.sort_values(["location_id", "date_for_month"])
    monthly["next_positive"] = monthly.groupby("location_id")["positive"].shift(-1)
    monthly["previous_cases"] = monthly.groupby("location_id")["cases"].shift(1)
    persistence_base = monthly[monthly["positive"] & monthly["next_positive"].notna()]
    persistence = float(persistence_base["next_positive"].mean()) if len(persistence_base) else 0.0
    growth_base = monthly[(monthly["previous_cases"] > 0) & (monthly["cases"] > 0)].copy()
    growth_ratio = float(np.median(growth_base["cases"] / growth_base["previous_cases"])) if len(growth_base) else 0.0
    decay_base = monthly[(monthly["previous_cases"] > 0) & (monthly["cases"] < monthly["previous_cases"])].copy()
    decay_ratio = float(np.median(decay_base["cases"] / decay_base["previous_cases"])) if len(decay_base) else 0.0

    run_lengths: list[int] = []
    spike_ratios: list[float] = []
    for _, group in monthly.groupby("location_id"):
        positives = group["positive"].astype(bool).tolist()
        run_lengths.extend(compute_run_lengths(positives))
        positive_cases = group.loc[group["cases"] > 0, "cases"]
        if len(positive_cases):
            median_positive = float(positive_cases.median())
            if median_positive > 0:
                spike_ratios.append(float(positive_cases.max() / median_positive))

    rows: list[dict[str, Any]] = []
    for _, row in month_features.iterrows():
        for metric in [
            "external_monthly_measles_seasonality_index",
            "external_monthly_positive_rate",
            "external_monthly_mean_log_cases",
            "external_monthly_p90_log_cases",
        ]:
            rows.append(
                {
                    "pattern_scope": "month",
                    "month": int(row["month"]),
                    "metric": metric,
                    "value": float(row[metric]),
                    "n_rows": int(by_month.loc[by_month["month"] == row["month"], "external_rows"].iloc[0]),
                    "notes": "Derived from public non-Ethiopian measles rows; used as support priors only.",
                }
            )
    global_metrics = {
        "external_rows_available": len(ext),
        "external_monthly_rows_used": len(monthly),
        "external_locations_used": monthly["location_id"].nunique(),
        "outbreak_persistence_probability": persistence,
        "median_positive_run_months": float(np.median(run_lengths)) if run_lengths else 0.0,
        "p90_positive_run_months": float(np.quantile(run_lengths, 0.9)) if run_lengths else 0.0,
        "median_spike_ratio": float(np.median(spike_ratios)) if spike_ratios else 0.0,
        "median_positive_growth_ratio": growth_ratio,
        "median_decay_ratio": decay_ratio,
    }
    for metric, value in global_metrics.items():
        rows.append(
            {
                "pattern_scope": "global",
                "month": np.nan,
                "metric": metric,
                "value": float(value),
                "n_rows": int(len(monthly)),
                "notes": "External pattern summary; not an Ethiopia ground-truth label.",
            }
        )

    metadata = {
        "external_path": str(external_path),
        "external_rows": int(len(ext)),
        "external_monthly_rows_used": int(len(monthly)),
        "external_locations_used": int(monthly["location_id"].nunique()),
    }
    return month_features, pd.DataFrame(rows), metadata


def add_external_priors(df: pd.DataFrame, month_features: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(month_features, on="month", how="left")
    for column in month_features.columns:
        if column != "month":
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out


def select_training_splits(df: pd.DataFrame, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labelled = df[df["target_outbreak_next_month"].notna()].copy()
    labelled["target_outbreak_next_month"] = labelled["target_outbreak_next_month"].astype(int)
    train = labelled[labelled["year"] < test_year].copy()
    test = labelled[labelled["year"] >= test_year].copy()
    threshold_fit = train[train["year"] < test_year - 1].copy()
    threshold_valid = train[train["year"] == test_year - 1].copy()
    if threshold_fit.empty or threshold_valid.empty or threshold_valid["target_outbreak_next_month"].nunique() < 2:
        threshold_fit = train.copy()
        threshold_valid = train.copy()
    if train.empty or test.empty:
        raise RuntimeError("Time split failed. Need both training rows before test year and test rows at/after test year.")
    return threshold_fit, threshold_valid, train, test


def choose_best_candidate(candidates: list[CandidateResult]) -> CandidateResult:
    def score(candidate: CandidateResult) -> tuple[float, float]:
        return (
            float(candidate.validation_metrics.get("f2") or 0.0),
            float(candidate.validation_metrics.get("pr_auc") or 0.0),
        )

    return sorted(candidates, key=score, reverse=True)[0]


def choose_primary_candidate(candidates: list[CandidateResult], primary_model: str) -> CandidateResult:
    normalized_primary = primary_model.strip().lower()
    for candidate in candidates:
        if candidate.name.lower() == normalized_primary:
            return candidate
    return choose_best_candidate(candidates)


def make_prediction_output(df: pd.DataFrame, result: CandidateResult) -> pd.DataFrame:
    X_all = result.preprocessor.transform(df)
    probabilities = result.model.predict_proba(X_all)
    output = df[
        [
            "location_id",
            "period_start",
            "prediction_period_start",
            "admin1_region",
            "admin2_zone",
            "admin3_woreda",
            "year",
            "month",
            "target_outbreak_next_month",
            "target_cases_next_month",
        ]
    ].copy()
    output["model_name"] = result.name
    output["risk_probability"] = probabilities
    output["risk_prediction"] = (probabilities >= result.threshold).astype(int)
    output["actual_label_available"] = output["target_outbreak_next_month"].notna().astype(int)
    output["risk_bucket"] = pd.cut(
        output["risk_probability"],
        bins=[-0.001, 0.25, 0.5, 0.75, 1.001],
        labels=["low", "moderate", "high", "very_high"],
    ).astype(str)
    return output


def make_new_2025_outbreak_locations(df: pd.DataFrame, top_latest: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "location_id",
        "admin1_region",
        "admin2_zone",
        "admin3_woreda",
        "period_start",
        "target_outbreak",
        "target_cases",
        "target_deaths",
    ]
    available = [column for column in base_columns if column in df.columns]
    base = df[available].copy()
    base["period_start"] = pd.to_datetime(base["period_start"])
    for column in ["target_outbreak", "target_cases", "target_deaths"]:
        if column not in base.columns:
            base[column] = 0
        base[column] = pd.to_numeric(base[column], errors="coerce").fillna(0)
    outbreaks = base[base["target_outbreak"].eq(1)].sort_values(["location_id", "period_start"])
    columns = [
        "location_id",
        "admin1_region",
        "admin2_zone",
        "admin3_woreda",
        "first_outbreak_month",
        "first_outbreak_cases",
        "first_outbreak_deaths",
        "outbreak_months_2025",
        "cases_2025",
        "deaths_2025",
        "previous_outbreak_count",
        "latest_prediction_period_start",
        "latest_risk_probability",
        "latest_risk_bucket",
        "latest_risk_prediction",
    ]
    if outbreaks.empty:
        return pd.DataFrame(columns=columns)

    first = outbreaks.groupby("location_id", as_index=False).first()
    first = first[first["period_start"].dt.year.eq(2025)].copy()
    if first.empty:
        return pd.DataFrame(columns=columns)

    rows_2025 = base[base["period_start"].dt.year.eq(2025)].copy()
    agg_2025 = (
        rows_2025.groupby("location_id", as_index=False)
        .agg(
            outbreak_months_2025=("target_outbreak", "sum"),
            cases_2025=("target_cases", "sum"),
            deaths_2025=("target_deaths", "sum"),
        )
    )
    previous = (
        base[base["period_start"].dt.year.lt(2025)]
        .groupby("location_id", as_index=False)
        .agg(previous_outbreak_count=("target_outbreak", "sum"))
    )
    latest = top_latest[
        [
            "location_id",
            "prediction_period_start",
            "risk_probability",
            "risk_bucket",
            "risk_prediction",
        ]
    ].rename(
        columns={
            "prediction_period_start": "latest_prediction_period_start",
            "risk_probability": "latest_risk_probability",
            "risk_bucket": "latest_risk_bucket",
            "risk_prediction": "latest_risk_prediction",
        }
    )
    out = (
        first.rename(
            columns={
                "period_start": "first_outbreak_month",
                "target_cases": "first_outbreak_cases",
                "target_deaths": "first_outbreak_deaths",
            }
        )
        .merge(agg_2025, on="location_id", how="left")
        .merge(previous, on="location_id", how="left")
        .merge(latest, on="location_id", how="left")
    )
    out["previous_outbreak_count"] = pd.to_numeric(out["previous_outbreak_count"], errors="coerce").fillna(0).astype(int)
    out = out[out["previous_outbreak_count"].eq(0)].copy()
    for column in ["first_outbreak_month", "latest_prediction_period_start"]:
        out[column] = pd.to_datetime(out[column], errors="coerce").dt.date.astype(str)
    return out[columns].sort_values(["first_outbreak_month", "cases_2025"], ascending=[True, False]).reset_index(drop=True)


def make_next_month_alert_woredas(top_latest: pd.DataFrame) -> pd.DataFrame:
    latest = top_latest.copy()
    high_sources = latest[latest["risk_bucket"].isin(["high", "very_high"])].copy()
    alert_columns = [
        "location_id",
        "prediction_period_start",
        "admin1_region",
        "admin2_zone",
        "admin3_woreda",
        "risk_probability",
        "risk_prediction",
        "risk_bucket",
        "alert_role",
        "alert_level",
        "source_zone_high_risk_count",
        "source_zone_very_high_count",
        "source_zone_max_risk_probability",
        "source_zone_top_source_woreda",
        "alert_reason",
    ]
    if high_sources.empty:
        return pd.DataFrame(columns=alert_columns)

    zone_summary = (
        high_sources.groupby(["admin1_region", "admin2_zone"], as_index=False)
        .agg(
            source_zone_high_risk_count=("location_id", "nunique"),
            source_zone_very_high_count=("risk_bucket", lambda s: int((s == "very_high").sum())),
            source_zone_max_risk_probability=("risk_probability", "max"),
        )
    )
    top_source = (
        high_sources.sort_values("risk_probability", ascending=False)
        .groupby(["admin1_region", "admin2_zone"], as_index=False)
        .first()[["admin1_region", "admin2_zone", "admin3_woreda"]]
        .rename(columns={"admin3_woreda": "source_zone_top_source_woreda"})
    )
    zone_summary = zone_summary.merge(top_source, on=["admin1_region", "admin2_zone"], how="left")
    alerts = latest.merge(zone_summary, on=["admin1_region", "admin2_zone"], how="inner")
    alerts["alert_role"] = np.where(alerts["risk_bucket"].isin(["high", "very_high"]), "high_risk_source", "nearby_same_zone")
    alerts["alert_level"] = np.select(
        [
            alerts["risk_bucket"].eq("very_high"),
            alerts["risk_bucket"].eq("high"),
            alerts["source_zone_very_high_count"].gt(0),
        ],
        ["critical", "high", "watch"],
        default="monitor",
    )
    alerts["alert_reason"] = np.where(
        alerts["alert_role"].eq("high_risk_source"),
        "This woreda is high or very-high risk for the next prediction month.",
        "This woreda is in the same zone as a high or very-high risk source woreda.",
    )
    order = {"critical": 0, "high": 1, "watch": 2, "monitor": 3}
    alerts["_alert_order"] = alerts["alert_level"].map(order).fillna(9)
    alerts = alerts.sort_values(["_alert_order", "risk_probability"], ascending=[True, False]).drop_duplicates("location_id")
    return alerts[alert_columns].reset_index(drop=True)


def feature_family(feature: str) -> str:
    if feature.startswith("admin1_region_"):
        return "region"
    if feature.startswith("admin2_zone_"):
        return "zone"
    if feature.startswith("admin3_woreda_"):
        return "woreda"
    if feature.startswith("external_"):
        return "external_prior"
    if "lag" in feature or "rolling" in feature:
        return "lagged_outbreak_signal"
    if "coverage" in feature or "dropout" in feature or "susceptible" in feature:
        return "immunity_proxy"
    if "rainfall" in feature or "dry_season" in feature:
        return "climate"
    if "food" in feature or "conflict" in feature:
        return "stress_proxy"
    if "population" in feature or "density" in feature:
        return "population"
    return "other"


def make_feature_importance(result: CandidateResult) -> pd.DataFrame:
    importance = result.model.feature_importance(result.feature_columns).copy()
    importance["importance"] = pd.to_numeric(importance["importance"], errors="coerce").fillna(0.0)
    total = float(importance["importance"].sum())
    if total > 0:
        importance["importance_share"] = importance["importance"] / total
    else:
        importance["importance_share"] = 0.0
    importance["feature_family"] = importance["feature"].map(feature_family)
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def confusion_matrix_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    cm = metrics["confusion_matrix"]
    return pd.DataFrame(
        [
            {"actual": 0, "predicted_0": cm["tn"], "predicted_1": cm["fp"]},
            {"actual": 1, "predicted_0": cm["fn"], "predicted_1": cm["tp"]},
        ]
    )


def model_comparison_frame(candidates: list[CandidateResult], primary_model: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        validation = candidate.validation_metrics
        test = candidate.test_metrics
        rows.append(
            {
                "model_name": candidate.name,
                "model_role": "primary" if candidate.name == primary_model else "comparison",
                "threshold": candidate.threshold,
                "validation_precision": validation.get("precision"),
                "validation_recall": validation.get("recall"),
                "validation_f1": validation.get("f1"),
                "validation_f2": validation.get("f2"),
                "validation_roc_auc": validation.get("roc_auc"),
                "validation_pr_auc": validation.get("pr_auc"),
                "test_precision": test.get("precision"),
                "test_recall": test.get("recall"),
                "test_f1": test.get("f1"),
                "test_f2": test.get("f2"),
                "test_roc_auc": test.get("roc_auc"),
                "test_pr_auc": test.get("pr_auc"),
                "test_tn": test.get("confusion_matrix", {}).get("tn"),
                "test_fp": test.get("confusion_matrix", {}).get("fp"),
                "test_fn": test.get("confusion_matrix", {}).get("fn"),
                "test_tp": test.get("confusion_matrix", {}).get("tp"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["model_role", "validation_f2", "validation_pr_auc"],
        ascending=[False, False, False],
    )


def extract_probability_model(result: CandidateResult) -> Any:
    model = result.model
    if hasattr(model, "model") and getattr(model, "model") is not None:
        return getattr(model, "model")
    return model


def save_model_artifact(
    path: Path,
    result: CandidateResult,
    metrics: dict[str, Any],
    month_features: pd.DataFrame,
) -> Path:
    artifact = {
        "model_name": result.name,
        "model": extract_probability_model(result),
        "threshold": float(result.threshold),
        "risk_bucket_bins": [-0.001, 0.25, 0.5, 0.75, 1.001],
        "risk_bucket_labels": ["low", "moderate", "high", "very_high"],
        "preprocessor_state": result.preprocessor.to_state(),
        "feature_columns": result.feature_columns,
        "month_features": month_features.to_dict(orient="records"),
        "metadata": {
            "model_goal": metrics.get("model_goal"),
            "input_file": metrics.get("input_file"),
            "external_file": metrics.get("external_file"),
            "horizon_months": metrics.get("horizon_months"),
            "test_year": metrics.get("test_year"),
            "decision_threshold": metrics.get("decision_threshold"),
            "selected_time_test_2025": metrics.get("selected_time_test_2025"),
            "created_by": "scripts/train_measles_mvp_model.py",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def write_report(
    report_path: Path,
    metrics: dict[str, Any],
    selected: CandidateResult,
    comparison: pd.DataFrame,
    feature_importance: pd.DataFrame,
    top_latest: pd.DataFrame,
    new_2025_outbreaks: pd.DataFrame,
    next_month_alerts: pd.DataFrame,
    external_metadata: dict[str, Any],
    training_mode: str,
) -> None:
    top_features = feature_importance.head(10)[["feature", "importance_share", "feature_family"]]
    input_file = str(metrics.get("input_file", ""))
    real_only_input = "real" in Path(input_file).stem.lower()
    limitation_text = (
        "This run used the real Ethiopia measles update line-list aggregation. "
        "Zero-filled panel months represent no line-list record for that woreda-month, not synthetic generated labels."
        if real_only_input
        else "The real EPHI measles rows are limited, so many Ethiopia training rows come from the removable MVP expansion layer."
    )
    lines = [
        "# Training Readiness Report",
        "",
        "## What was trained",
        "",
        "The MVP model predicts next-month measles outbreak risk for Ethiopia woreda-month rows.",
        "The current-month target columns are excluded from the feature matrix to avoid direct leakage.",
        "XGBoost is the primary model for the final project; Random Forest and KNN are comparison models.",
        "",
        f"- Primary model: `{selected.name}`",
        f"- Training backend: `{training_mode}`",
        f"- Decision threshold: `{selected.threshold:.2f}`",
        f"- Test rows: `{selected.test_metrics['rows']}`",
        f"- Test positive rows: `{selected.test_metrics['positive_rows']}`",
        "",
        "## Time-Based 2025 Test Metrics",
        "",
        f"- Precision: `{selected.test_metrics['precision']:.3f}`",
        f"- Recall: `{selected.test_metrics['recall']:.3f}`",
        f"- F1: `{selected.test_metrics['f1']:.3f}`",
        f"- F2: `{selected.test_metrics['f2']:.3f}`",
        f"- ROC-AUC: `{selected.test_metrics['roc_auc']:.3f}`" if selected.test_metrics["roc_auc"] is not None else "- ROC-AUC: `n/a`",
        f"- PR-AUC: `{selected.test_metrics['pr_auc']:.3f}`" if selected.test_metrics["pr_auc"] is not None else "- PR-AUC: `n/a`",
        "",
        "## Model Comparison",
        "",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"- `{row['model_name']}` ({row['model_role']}): "
            f"test F2 `{row['test_f2']:.3f}`, recall `{row['test_recall']:.3f}`, "
            f"precision `{row['test_precision']:.3f}`"
        )
    lines.extend(
        [
            "",
            "## Defense Outputs",
            "",
            f"- `top_risk_woredas_latest.csv`: `{len(top_latest)}` latest prediction rows sorted by risk.",
            f"- `new_2025_outbreak_locations.csv`: `{len(new_2025_outbreaks)}` woredas whose first recorded outbreak occurred in 2025.",
            f"- `next_month_alert_woredas.csv`: `{len(next_month_alerts)}` high-risk or same-zone nearby alert rows.",
            "",
            "## External Resource Use",
            "",
            "External-country measles rows were not appended to Ethiopia evaluation.",
            "They were summarized into monthly seasonality and outbreak-shape priors only.",
            "",
            f"- External source file: `{external_metadata.get('external_path')}`",
            f"- External rows available: `{external_metadata.get('external_rows', 0)}`",
            f"- External monthly rows used: `{external_metadata.get('external_monthly_rows_used', 0)}`",
            f"- External locations used: `{external_metadata.get('external_locations_used', 0)}`",
            "",
            "## Top Model Drivers",
            "",
        ]
    )
    for _, row in top_features.iterrows():
        lines.append(f"- `{row['feature']}`: {row['importance_share']:.3f} ({row['feature_family']})")
    lines.extend(
        [
            "",
            "## Latest High-Risk Woredas",
            "",
        ]
    )
    for _, row in top_latest.head(10).iterrows():
        lines.append(
            f"- {row['admin3_woreda']} ({row['admin1_region']}): "
            f"{row['risk_probability']:.3f} risk for {pd.to_datetime(row['prediction_period_start']).date()}"
        )
    lines.extend(
        [
            "",
            "## What this can support",
            "",
            "This is suitable for a defense demo, feature-importance explanation, and dashboard risk ranking.",
            "It is strongest as a prototype decision-support model, not as an official national surveillance result.",
            "",
            "## Important limitations",
            "",
            limitation_text,
            "Public external-country rows support outbreak behavior assumptions, but they are not Ethiopian ground truth.",
            "Final academic claims should clearly state the label source and any zero-fill/imputation assumptions.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_ethiopia_matrix(args.input, args.horizon_months)

    external_path = None if args.no_external_priors else resolve_external_path(args.external)
    month_features, external_patterns, external_metadata = build_external_patterns(external_path)
    df = add_external_priors(df, month_features)

    feature_leaks = sorted((LEAKAGE_COLUMNS | PROVENANCE_OR_FLAG_COLUMNS).intersection(set(df.columns)))
    threshold_fit, threshold_valid, train, test = select_training_splits(df, args.test_year)
    if sklearn_available():
        builders = make_sklearn_builders(args.random_state)
        training_mode = "scikit-learn"
    else:
        builders = make_fallback_builders(args.random_state)
        training_mode = "numpy-pandas fallback"

    candidates: list[CandidateResult] = []
    for name, builder in builders.items():
        candidates.append(fit_candidate(name, builder, threshold_fit, threshold_valid, train, test))
    selected = choose_primary_candidate(candidates, args.primary_model)
    best_by_validation = choose_best_candidate(candidates)
    comparison = model_comparison_frame(candidates, selected.name)

    predictions = make_prediction_output(df, selected)
    latest_period = predictions["prediction_period_start"].max()
    top_latest = (
        predictions[predictions["prediction_period_start"].eq(latest_period)]
        .sort_values("risk_probability", ascending=False)
        .reset_index(drop=True)
    )
    new_2025_outbreaks = make_new_2025_outbreak_locations(df, top_latest)
    next_month_alerts = make_next_month_alert_woredas(top_latest)
    feature_importance = make_feature_importance(selected)
    stratified_metrics = stratified_validation(
        df[df["target_outbreak_next_month"].notna()].copy(),
        builders[selected.name],
        args.random_state,
    )

    metrics = {
        "model_goal": "Predict next-month Ethiopia woreda measles outbreak risk.",
        "selected_model": selected.name,
        "primary_model_requested": args.primary_model,
        "primary_model_used": selected.name,
        "best_validation_model": best_by_validation.name,
        "selection_policy": "Use the requested primary model when available; keep Random Forest and KNN as comparison models.",
        "training_mode": training_mode,
        "decision_threshold": selected.threshold,
        "input_file": str(args.input),
        "external_file": external_metadata.get("external_path"),
        "horizon_months": args.horizon_months,
        "test_year": args.test_year,
        "rows": {
            "ethiopia_rows_total": int(len(df)),
            "labelled_rows_for_training_or_testing": int(df["target_outbreak_next_month"].notna().sum()),
            "threshold_fit_rows": int(len(threshold_fit)),
            "threshold_validation_rows": int(len(threshold_valid)),
            "time_train_rows": int(len(train)),
            "time_test_rows": int(len(test)),
            "external_rows_directly_in_training": 0,
        },
        "defense_outputs": {
            "top_risk_woredas_latest_rows": int(len(top_latest)),
            "new_2025_outbreak_locations_rows": int(len(new_2025_outbreaks)),
            "next_month_alert_woredas_rows": int(len(next_month_alerts)),
        },
        "class_balance": {
            "train_positive_rate": float(train["target_outbreak_next_month"].mean()),
            "test_positive_rate": float(test["target_outbreak_next_month"].mean()),
        },
        "leakage_columns_excluded": feature_leaks,
        "candidate_metrics": {
            candidate.name: {
                "validation": candidate.validation_metrics,
                "time_test_2025": candidate.test_metrics,
            }
            for candidate in candidates
        },
        "selected_time_test_2025": selected.test_metrics,
        "selected_stratified_demo_validation": stratified_metrics,
        "external_patterns": external_metadata,
        "notes": [
            "External country rows are summarized into priors only; they are not Ethiopia labels.",
            "Current target_cases, target_deaths, and target_outbreak are excluded from features.",
            "Lagged target features are allowed because they represent previous-month surveillance signals.",
        ],
    }

    predictions.to_csv(args.output_dir / "measles_next_month_predictions.csv", index=False)
    top_latest.to_csv(args.output_dir / "top_risk_woredas_latest.csv", index=False)
    new_2025_outbreaks.to_csv(args.output_dir / "new_2025_outbreak_locations.csv", index=False)
    next_month_alerts.to_csv(args.output_dir / "next_month_alert_woredas.csv", index=False)
    feature_importance.to_csv(args.output_dir / "feature_importance.csv", index=False)
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    external_patterns.to_csv(args.output_dir / "external_outbreak_patterns.csv", index=False)
    confusion_matrix_frame(selected.test_metrics).to_csv(args.output_dir / "confusion_matrix.csv", index=False)
    artifact_path = save_model_artifact(args.output_dir / "xgboost_model_artifact.joblib", selected, metrics, month_features)
    (args.output_dir / "evaluation_metrics.json").write_text(
        json.dumps(json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    write_report(
        args.report,
        metrics,
        selected,
        comparison,
        feature_importance,
        top_latest,
        new_2025_outbreaks,
        next_month_alerts,
        external_metadata,
        training_mode,
    )

    print(json.dumps(json_safe({
        "primary_model": selected.name,
        "best_validation_model": best_by_validation.name,
        "training_mode": training_mode,
        "test_f2": selected.test_metrics["f2"],
        "test_recall": selected.test_metrics["recall"],
        "test_precision": selected.test_metrics["precision"],
        "new_2025_outbreak_locations": len(new_2025_outbreaks),
        "next_month_alert_woredas": len(next_month_alerts),
        "artifact": str(artifact_path),
        "outputs": str(args.output_dir),
    }), indent=2))


if __name__ == "__main__":
    main()
