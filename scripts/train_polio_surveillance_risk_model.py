"""Train polio AFP surveillance-risk alerting models.

This module trains derived-label AFP surveillance models. It does not train a
confirmed polio outbreak model because confirmed WPV/cVDPV labels are not
available in the current AFP dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "polio_afp_woreda_month_features.csv"
DEFAULT_TRAINING_MATRIX = ROOT / "data" / "processed" / "polio_surveillance_training_matrix.csv"
DEFAULT_OUTPUT_DIR = ROOT / "model_outputs_polio_afp"
DEFAULT_REPORT = ROOT / "reports" / "polio_surveillance_training_report.md"
RANDOM_STATE = 42

CURRENT_FLAG_COLUMNS = [
    "high_surveillance_risk_flag",
    "poor_stool_adequacy_flag",
    "delayed_reporting_flag",
    "under_vaccinated_afp_signal",
    "suspected_poliovirus_signal",
]

TARGET_COLUMNS = [
    "target_high_surveillance_risk_next_month",
    "target_poor_stool_adequacy_next_month",
    "target_delayed_reporting_next_month",
    "target_under_vaccinated_afp_signal_next_month",
    "target_suspected_poliovirus_signal_next_month",
]

TARGET_DISPLAY = {
    "target_high_surveillance_risk_next_month": "high_surveillance_risk",
    "target_poor_stool_adequacy_next_month": "poor_stool_adequacy",
    "target_delayed_reporting_next_month": "delayed_reporting",
    "target_under_vaccinated_afp_signal_next_month": "under_vaccinated_afp",
    "target_suspected_poliovirus_signal_next_month": "suspected_poliovirus",
}

PROBABILITY_COLUMNS = {
    "target_high_surveillance_risk_next_month": "high_surveillance_risk_probability",
    "target_poor_stool_adequacy_next_month": "poor_stool_adequacy_probability",
    "target_delayed_reporting_next_month": "delayed_reporting_probability",
    "target_under_vaccinated_afp_signal_next_month": "under_vaccinated_afp_probability",
    "target_suspected_poliovirus_signal_next_month": "suspected_poliovirus_probability",
}

META_COLUMNS = {
    "location_id",
    "period_start",
    "prediction_period_start",
    "admin1_region",
    "admin2_zone",
    "admin3_woreda",
    "admin3_pcode",
}

TEXT_COLUMNS = {
    "country",
    "data_level",
    "source_notes",
    "risk_bucket",
}

CATEGORICAL_COLUMNS = ["admin1_region", "admin2_zone"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--training-matrix", type=Path, default=DEFAULT_TRAINING_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


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
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(path.stem + "_new" + path.suffix)
        df.to_csv(fallback, index=False)
        return fallback


def load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing polio AFP feature file: {path}")
    df = pd.read_csv(path)
    required = {
        "location_id",
        "period_start",
        "admin1_region",
        "admin2_zone",
        "admin3_woreda",
        "afp_cases",
        "adequate_stool_rate",
        "timely_notification_rate",
        "timely_investigation_rate",
        "timely_lab_result_rate",
        "median_notification_delay_days",
        "median_investigation_delay_days",
        "median_lab_result_delay_days",
        "under_vaccinated_afp_cases",
        "suspected_poliovirus_lab_result_count",
        "polio_surveillance_risk_score",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["period_start", "location_id"]).copy()
    df["year"] = df["period_start"].dt.year
    df["month"] = df["period_start"].dt.month
    return df.sort_values(["location_id", "period_start"]).reset_index(drop=True)


def add_current_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    afp_cases = pd.to_numeric(out["afp_cases"], errors="coerce").fillna(0)
    out["high_surveillance_risk_flag"] = (pd.to_numeric(out["polio_surveillance_risk_score"], errors="coerce").fillna(0) >= 50).astype(int)
    out["poor_stool_adequacy_flag"] = ((afp_cases > 0) & (pd.to_numeric(out["adequate_stool_rate"], errors="coerce").fillna(0) < 0.80)).astype(int)
    out["delayed_reporting_flag"] = (
        (afp_cases > 0)
        & (
            (pd.to_numeric(out["timely_notification_rate"], errors="coerce").fillna(0) < 0.80)
            | (pd.to_numeric(out["timely_investigation_rate"], errors="coerce").fillna(0) < 0.80)
            | (pd.to_numeric(out["timely_lab_result_rate"], errors="coerce").fillna(0) < 0.80)
            | (pd.to_numeric(out["median_notification_delay_days"], errors="coerce").fillna(0) > 7)
            | (pd.to_numeric(out["median_investigation_delay_days"], errors="coerce").fillna(0) > 2)
            | (pd.to_numeric(out["median_lab_result_delay_days"], errors="coerce").fillna(0) > 14)
        )
    ).astype(int)
    out["under_vaccinated_afp_signal"] = (pd.to_numeric(out["under_vaccinated_afp_cases"], errors="coerce").fillna(0) > 0).astype(int)
    out["suspected_poliovirus_signal"] = (pd.to_numeric(out["suspected_poliovirus_lab_result_count"], errors="coerce").fillna(0) > 0).astype(int)
    return out


def add_next_month_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prediction_period_start"] = out["period_start"] + pd.offsets.MonthBegin(1)
    future = out[["location_id", "period_start", *CURRENT_FLAG_COLUMNS]].copy()
    future = future.rename(
        columns={
            "period_start": "prediction_period_start",
            "high_surveillance_risk_flag": "target_high_surveillance_risk_next_month",
            "poor_stool_adequacy_flag": "target_poor_stool_adequacy_next_month",
            "delayed_reporting_flag": "target_delayed_reporting_next_month",
            "under_vaccinated_afp_signal": "target_under_vaccinated_afp_signal_next_month",
            "suspected_poliovirus_signal": "target_suspected_poliovirus_signal_next_month",
        }
    )
    return out.merge(future, on=["location_id", "prediction_period_start"], how="left")


class FeaturePreprocessor:
    def __init__(self) -> None:
        self.feature_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.medians: pd.Series | None = None

    def fit_transform(self, df: pd.DataFrame, target_column: str) -> np.ndarray:
        matrix = self._build_matrix(df, target_column, fit=True)
        self.feature_columns = list(matrix.columns)
        return matrix.to_numpy(dtype=float)

    def transform(self, df: pd.DataFrame, target_column: str) -> np.ndarray:
        matrix = self._build_matrix(df, target_column, fit=False)
        matrix = matrix.reindex(columns=self.feature_columns, fill_value=0.0)
        return matrix.to_numpy(dtype=float)

    def _build_matrix(self, df: pd.DataFrame, target_column: str, fit: bool) -> pd.DataFrame:
        excluded = set(TARGET_COLUMNS) | set(CURRENT_FLAG_COLUMNS) | META_COLUMNS | TEXT_COLUMNS
        candidate_columns = [column for column in df.columns if column not in excluded]
        categorical = [column for column in CATEGORICAL_COLUMNS if column in candidate_columns]
        numeric = [
            column
            for column in candidate_columns
            if column not in categorical and pd.api.types.is_numeric_dtype(df[column])
        ]
        if fit:
            self.numeric_columns = numeric
            self.categorical_columns = categorical
            numeric_frame = df[numeric].apply(pd.to_numeric, errors="coerce")
            self.medians = numeric_frame.replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
        if self.medians is None:
            raise RuntimeError("FeaturePreprocessor must be fit before transform.")
        numeric_frame = (
            df[self.numeric_columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self.medians)
        )
        categorical_frame = df[self.categorical_columns].fillna("missing").astype(str)
        if categorical:
            dummies = pd.get_dummies(categorical_frame, prefix=categorical, dtype=float)
            return pd.concat([numeric_frame.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        return numeric_frame.reset_index(drop=True)

    def to_state(self) -> dict[str, Any]:
        if self.medians is None:
            raise RuntimeError("FeaturePreprocessor must be fit before export.")
        return {
            "feature_columns": self.feature_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "medians": {str(key): float(value) for key, value in self.medians.to_dict().items()},
        }


class ConstantProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p1 = np.full(len(x), self.probability, dtype=float)
        return np.vstack([1 - p1, p1]).T

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.array([], dtype=float)


@dataclass
class TargetResult:
    target_column: str
    model: Any
    preprocessor: FeaturePreprocessor
    threshold: float
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    feature_importance: pd.DataFrame


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
    return float((pos_rank_sum - (pos * (pos + 1) / 2.0)) / (pos * neg))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos = int(np.sum(y_true == 1))
    if pos == 0:
        return None
    order = np.argsort(-scores)
    sorted_y = y_true[order]
    tp = np.cumsum(sorted_y == 1)
    precision = tp / np.maximum(np.arange(1, len(sorted_y) + 1), 1)
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
    if len(y_true) == 0 or int(np.sum(y_true == 1)) == 0:
        return 0.5
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        score = compute_metrics(y_true, probabilities, float(threshold))["f2"]
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def build_xgboost_model(random_state: int, y_train: np.ndarray) -> Any:
    if len(np.unique(y_train)) < 2:
        return ConstantProbabilityModel(float(np.mean(y_train)) if len(y_train) else 0.0)
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=160,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )
    except Exception:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )


def positive_rate(frame: pd.DataFrame, target_column: str) -> float | None:
    series = frame[target_column].dropna()
    return float(series.mean()) if len(series) else None


def fit_target(
    target_column: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    random_state: int,
) -> TargetResult:
    train_labelled = train.dropna(subset=[target_column]).copy()
    validation_labelled = validation.dropna(subset=[target_column]).copy()
    test_labelled = test.dropna(subset=[target_column]).copy()
    preprocessor = FeaturePreprocessor()
    x_train = preprocessor.fit_transform(train_labelled, target_column)
    y_train = train_labelled[target_column].astype(int).to_numpy()
    model = build_xgboost_model(random_state, y_train)
    if isinstance(model, ConstantProbabilityModel):
        model.fit = None
    else:
        model.fit(x_train, y_train, sample_weight=class_balanced_weights(y_train))

    x_validation = preprocessor.transform(validation_labelled, target_column)
    y_validation = validation_labelled[target_column].astype(int).to_numpy()
    validation_probabilities = model.predict_proba(x_validation)[:, 1] if len(validation_labelled) else np.array([])
    threshold = tune_threshold(y_validation, validation_probabilities)
    validation_metrics = compute_metrics(y_validation, validation_probabilities, threshold) if len(validation_labelled) else {}

    x_test = preprocessor.transform(test_labelled, target_column)
    y_test = test_labelled[target_column].astype(int).to_numpy()
    test_probabilities = model.predict_proba(x_test)[:, 1] if len(test_labelled) else np.array([])
    test_metrics = compute_metrics(y_test, test_probabilities, threshold) if len(test_labelled) else {}

    importances = getattr(model, "feature_importances_", np.array([]))
    if len(importances) != len(preprocessor.feature_columns):
        importance = np.zeros(len(preprocessor.feature_columns), dtype=float)
    else:
        importance = np.asarray(importances, dtype=float)
    feature_importance = pd.DataFrame(
        {
            "target_name": TARGET_DISPLAY[target_column],
            "feature": preprocessor.feature_columns,
            "importance": importance,
        }
    )
    total = float(feature_importance["importance"].sum())
    feature_importance["importance_share"] = feature_importance["importance"] / total if total > 0 else 0.0
    return TargetResult(
        target_column=target_column,
        model=model,
        preprocessor=preprocessor,
        threshold=threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        feature_importance=feature_importance.sort_values("importance", ascending=False).reset_index(drop=True),
    )


def add_predictions(df: pd.DataFrame, results: list[TargetResult]) -> pd.DataFrame:
    out = df.copy()
    for result in results:
        x_all = result.preprocessor.transform(out, result.target_column)
        out[PROBABILITY_COLUMNS[result.target_column]] = result.model.predict_proba(x_all)[:, 1]
        out[PROBABILITY_COLUMNS[result.target_column].replace("_probability", "_prediction")] = (
            out[PROBABILITY_COLUMNS[result.target_column]] >= result.threshold
        ).astype(int)
    return out


def split_frame(df: pd.DataFrame, train_end_year: int, validation_year: int, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_year = df["prediction_period_start"].dt.year
    train = df[prediction_year.between(2021, train_end_year, inclusive="both")].copy()
    validation = df[prediction_year.eq(validation_year)].copy()
    test = df[prediction_year.eq(test_year)].copy()
    return train, validation, test


def reason_and_action(row: pd.Series, thresholds: dict[str, float]) -> tuple[str, str]:
    reasons: list[str] = []
    actions: list[str] = []
    if (
        row["suspected_poliovirus_probability"] >= thresholds["target_suspected_poliovirus_signal_next_month"]
        or pd.to_numeric(row.get("suspected_poliovirus_lab_result_count", 0), errors="coerce") > 0
    ):
        reasons.append("suspected poliovirus surveillance signal")
        actions.append("escalate surveillance review without treating it as confirmed outbreak")
    if (
        row["poor_stool_adequacy_probability"] >= thresholds["target_poor_stool_adequacy_next_month"]
        or pd.to_numeric(row.get("adequate_stool_rate", 1), errors="coerce") < 0.80
    ):
        reasons.append("poor stool adequacy risk")
        actions.append("verify stool collection and specimen handling")
    if (
        row["delayed_reporting_probability"] >= thresholds["target_delayed_reporting_next_month"]
        or pd.to_numeric(row.get("timely_notification_rate", 1), errors="coerce") < 0.80
        or pd.to_numeric(row.get("timely_investigation_rate", 1), errors="coerce") < 0.80
    ):
        reasons.append("delayed AFP notification or investigation risk")
        actions.append("strengthen AFP notification and rapid investigation follow-up")
    if pd.to_numeric(row.get("median_lab_result_delay_days", 0), errors="coerce") > 14:
        reasons.append("lab result delay risk")
        actions.append("follow up lab reporting pipeline")
    if (
        row["under_vaccinated_afp_probability"] >= thresholds["target_under_vaccinated_afp_signal_next_month"]
        or pd.to_numeric(row.get("under_vaccinated_afp_cases", 0), errors="coerce") > 0
    ):
        reasons.append("under-vaccinated AFP signal")
        actions.append("prioritize immunization outreach")
    if row["high_surveillance_risk_probability"] >= thresholds["target_high_surveillance_risk_next_month"]:
        reasons.append("high next-month surveillance-risk probability")
    if not reasons:
        reasons.append("moderate surveillance-risk watch signal")
        actions.append("continue AFP surveillance monitoring")
    actions = list(dict.fromkeys(actions))
    return "; ".join(dict.fromkeys(reasons)), "; ".join(actions)


def make_latest_alerts(predictions: pd.DataFrame, thresholds: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    latest_period = predictions["prediction_period_start"].max()
    latest = predictions[predictions["prediction_period_start"].eq(latest_period)].copy()
    if latest.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    high_threshold = thresholds["target_high_surveillance_risk_next_month"]
    suspected_threshold = thresholds["target_suspected_poliovirus_signal_next_month"]
    under_threshold = thresholds["target_under_vaccinated_afp_signal_next_month"]
    poor_threshold = thresholds["target_poor_stool_adequacy_next_month"]
    delayed_threshold = thresholds["target_delayed_reporting_next_month"]

    latest["_own_critical"] = (
        (latest["suspected_poliovirus_probability"] >= suspected_threshold)
        | (latest["high_surveillance_risk_probability"] >= 0.80)
    )
    latest["_own_high"] = (
        (latest["high_surveillance_risk_probability"] >= high_threshold)
        | (latest["under_vaccinated_afp_probability"] >= under_threshold)
    ) & ~latest["_own_critical"]
    sources = latest[latest["_own_critical"] | latest["_own_high"]].copy()

    zone_summary = pd.DataFrame()
    if not sources.empty:
        zone_summary = (
            sources.groupby(["admin1_region", "admin2_zone"], as_index=False)
            .agg(
                source_zone_high_risk_count=("location_id", "nunique"),
                source_zone_max_high_surveillance_risk_probability=("high_surveillance_risk_probability", "max"),
            )
        )
        top_sources = (
            sources.sort_values(["_own_critical", "high_surveillance_risk_probability"], ascending=[False, False])
            .groupby(["admin1_region", "admin2_zone"], as_index=False)
            .first()[["admin1_region", "admin2_zone", "admin3_woreda"]]
            .rename(columns={"admin3_woreda": "source_zone_top_source_woreda"})
        )
        zone_summary = zone_summary.merge(top_sources, on=["admin1_region", "admin2_zone"], how="left")
        latest = latest.merge(zone_summary, on=["admin1_region", "admin2_zone"], how="left")
    else:
        latest["source_zone_high_risk_count"] = 0
        latest["source_zone_max_high_surveillance_risk_probability"] = 0.0
        latest["source_zone_top_source_woreda"] = ""

    latest["source_zone_high_risk_count"] = latest["source_zone_high_risk_count"].fillna(0).astype(int)
    latest["source_zone_max_high_surveillance_risk_probability"] = latest["source_zone_max_high_surveillance_risk_probability"].fillna(0.0)
    latest["source_zone_top_source_woreda"] = latest["source_zone_top_source_woreda"].fillna("")
    latest["_zone_watch"] = latest["source_zone_high_risk_count"].gt(0) & ~latest["_own_critical"] & ~latest["_own_high"]
    latest["_monitor"] = (
        (latest["poor_stool_adequacy_probability"] >= poor_threshold)
        | (latest["delayed_reporting_probability"] >= delayed_threshold)
        | (latest["high_surveillance_risk_probability"] >= 0.25)
        | (latest["under_vaccinated_afp_probability"] >= 0.35)
        | (latest["suspected_poliovirus_probability"] >= 0.20)
    ) & ~latest["_own_critical"] & ~latest["_own_high"] & ~latest["_zone_watch"]
    latest["alert_level"] = np.select(
        [latest["_own_critical"], latest["_own_high"], latest["_zone_watch"], latest["_monitor"]],
        ["critical", "high", "watch", "monitor"],
        default="none",
    )
    latest["alert_role"] = np.select(
        [latest["_own_critical"] | latest["_own_high"], latest["_zone_watch"], latest["_monitor"]],
        ["high_risk_source", "nearby_same_zone", "monitor_signal"],
        default="none",
    )
    reason_action = latest.apply(lambda row: reason_and_action(row, thresholds), axis=1)
    latest["alert_reasons"] = [item[0] for item in reason_action]
    latest["recommended_action"] = [item[1] for item in reason_action]

    alert_columns = [
        "location_id",
        "prediction_period_start",
        "admin1_region",
        "admin2_zone",
        "admin3_woreda",
        "high_surveillance_risk_probability",
        "poor_stool_adequacy_probability",
        "delayed_reporting_probability",
        "under_vaccinated_afp_probability",
        "suspected_poliovirus_probability",
        "alert_level",
        "alert_role",
        "source_zone_high_risk_count",
        "source_zone_max_high_surveillance_risk_probability",
        "source_zone_top_source_woreda",
        "alert_reasons",
        "recommended_action",
    ]
    order = {"critical": 0, "high": 1, "watch": 2, "monitor": 3, "none": 4}
    latest["_alert_order"] = latest["alert_level"].map(order).fillna(9)
    alerts = (
        latest[latest["alert_level"].ne("none")]
        .sort_values(["_alert_order", "high_surveillance_risk_probability"], ascending=[True, False])
        .reset_index(drop=True)
    )
    zone_watch = alerts[alerts["alert_level"].eq("watch")].copy().reset_index(drop=True)
    recommendations = alerts[alert_columns].copy()
    return alerts[alert_columns], zone_watch[alert_columns], recommendations


def make_signal_trends(predictions: pd.DataFrame) -> pd.DataFrame:
    trend = (
        predictions.groupby("period_start", as_index=False)
        .agg(
            afp_cases=("afp_cases", "sum"),
            high_surveillance_risk_rows=("high_surveillance_risk_flag", "sum"),
            poor_stool_adequacy_rows=("poor_stool_adequacy_flag", "sum"),
            delayed_reporting_rows=("delayed_reporting_flag", "sum"),
            under_vaccinated_afp_rows=("under_vaccinated_afp_signal", "sum"),
            suspected_poliovirus_signal_rows=("suspected_poliovirus_signal", "sum"),
            mean_afp_surveillance_quality_score=("afp_surveillance_quality_score", "mean"),
            mean_polio_surveillance_risk_score=("polio_surveillance_risk_score", "mean"),
            total_under_vaccinated_afp_cases=("under_vaccinated_afp_cases", "sum"),
            total_suspected_poliovirus_lab_result_count=("suspected_poliovirus_lab_result_count", "sum"),
        )
        .sort_values("period_start")
    )
    trend["period_start"] = pd.to_datetime(trend["period_start"]).dt.date.astype(str)
    return trend


def make_model_comparison(results: list[TargetResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        validation = result.validation_metrics
        test = result.test_metrics
        rows.append(
            {
                "model_name": "xgboost",
                "target_name": TARGET_DISPLAY[result.target_column],
                "target_column": result.target_column,
                "threshold": result.threshold,
                "validation_rows": validation.get("rows"),
                "validation_positive_rows": validation.get("positive_rows"),
                "validation_precision": validation.get("precision"),
                "validation_recall": validation.get("recall"),
                "validation_f2": validation.get("f2"),
                "validation_roc_auc": validation.get("roc_auc"),
                "validation_pr_auc": validation.get("pr_auc"),
                "test_rows": test.get("rows"),
                "test_positive_rows": test.get("positive_rows"),
                "test_precision": test.get("precision"),
                "test_recall": test.get("recall"),
                "test_f2": test.get("f2"),
                "test_roc_auc": test.get("roc_auc"),
                "test_pr_auc": test.get("pr_auc"),
            }
        )
    return pd.DataFrame(rows)


def make_confusion_matrix(results: list[TargetResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        for split, metrics in [("validation_2024", result.validation_metrics), ("test_2025", result.test_metrics)]:
            cm = metrics.get("confusion_matrix", {})
            rows.append(
                {
                    "target_name": TARGET_DISPLAY[result.target_column],
                    "target_column": result.target_column,
                    "split": split,
                    "actual_0_predicted_0": cm.get("tn"),
                    "actual_0_predicted_1": cm.get("fp"),
                    "actual_1_predicted_0": cm.get("fn"),
                    "actual_1_predicted_1": cm.get("tp"),
                    "threshold": metrics.get("threshold"),
                }
            )
    return pd.DataFrame(rows)


def write_report(path: Path, metrics: dict[str, Any], comparison: pd.DataFrame, feature_importance: pd.DataFrame, alerts: pd.DataFrame) -> None:
    lines = [
        "# Polio AFP Surveillance-Risk Training Report",
        "",
        "## What was trained",
        "",
        "This run trained derived-label AFP surveillance-risk alerting models, not a confirmed polio outbreak model.",
        "",
        f"- Input rows: `{metrics['rows']['input_rows']}`",
        f"- Training/calibration rows: `{metrics['rows']['train_rows']}`",
        f"- Validation rows: `{metrics['rows']['validation_rows']}`",
        f"- Test/demo rows: `{metrics['rows']['test_rows']}`",
        f"- Latest alert rows: `{metrics['outputs']['latest_alert_rows']}`",
        "",
        "## Split Definition",
        "",
        "- Training/calibration: prediction years `2021-2023`",
        "- Validation: prediction year `2024`",
        "- Testing/demo: prediction year `2025`",
        "",
        "## Target Positive Rates",
        "",
    ]
    for target_name, values in metrics["target_positive_rates"].items():
        lines.append(
            f"- `{target_name}`: train `{values['train']}`, validation `{values['validation']}`, test `{values['test']}`"
        )
    lines.extend(["", "## 2025 Test Metrics", ""])
    for _, row in comparison.iterrows():
        lines.append(
            f"- `{row['target_name']}`: F2 `{row['test_f2']:.3f}`, recall `{row['test_recall']:.3f}`, "
            f"precision `{row['test_precision']:.3f}`, PR-AUC `{row['test_pr_auc']:.3f}`"
            if pd.notna(row["test_pr_auc"])
            else f"- `{row['target_name']}`: F2 `{row['test_f2']:.3f}`, recall `{row['test_recall']:.3f}`, precision `{row['test_precision']:.3f}`"
        )
    lines.extend(["", "## Top Features", ""])
    for target_name in comparison["target_name"].tolist():
        top = feature_importance[feature_importance["target_name"].eq(target_name)].head(5)
        for _, row in top.iterrows():
            lines.append(f"- `{target_name}` / `{row['feature']}`: `{row['importance_share']:.3f}`")
    lines.extend(
        [
            "",
            "## Latest Alerts",
            "",
        ]
    )
    for _, row in alerts.head(10).iterrows():
        lines.append(
            f"- `{row['alert_level']}`: {row['admin3_woreda']}, {row['admin2_zone']}, {row['admin1_region']} "
            f"for {pd.to_datetime(row['prediction_period_start']).date()} - {row['alert_reasons']}"
        )
    lines.extend(
        [
            "",
            "## Important Limitation",
            "",
            "These are AFP surveillance-risk and preparedness alerts. They should not be presented as confirmed polio outbreak probabilities unless official WPV/cVDPV positive labels are added.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_artifact(path: Path, results: list[TargetResult], metrics: dict[str, Any]) -> Path:
    artifact = {
        "model_goal": "Predict next-month AFP surveillance-risk and preparedness alert signals.",
        "models": {
            result.target_column: {
                "model": result.model,
                "threshold": result.threshold,
                "preprocessor_state": result.preprocessor.to_state(),
                "probability_column": PROBABILITY_COLUMNS[result.target_column],
                "target_display": TARGET_DISPLAY[result.target_column],
            }
            for result in results
        },
        "metrics": metrics,
        "limitation": "Derived AFP surveillance-risk model; not a confirmed polio outbreak model.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_input(args.input)
    df = add_current_flags(df)
    df = add_next_month_targets(df)
    safe_to_csv(df, args.training_matrix)

    train, validation, test = split_frame(df, args.train_end_year, args.validation_year, args.test_year)
    results = [fit_target(target, train, validation, test, args.random_state) for target in TARGET_COLUMNS]
    predictions = add_predictions(df, results)

    thresholds = {result.target_column: float(result.threshold) for result in results}
    latest_alerts, zone_watch, recommendations = make_latest_alerts(predictions, thresholds)
    signal_trends = make_signal_trends(predictions)
    comparison = make_model_comparison(results)
    feature_importance = pd.concat([result.feature_importance for result in results], ignore_index=True)
    confusion = make_confusion_matrix(results)

    prediction_cols = [
        "location_id",
        "period_start",
        "prediction_period_start",
        "admin1_region",
        "admin2_zone",
        "admin3_woreda",
        "year",
        "month",
        *TARGET_COLUMNS,
        *PROBABILITY_COLUMNS.values(),
        "high_surveillance_risk_prediction",
        "poor_stool_adequacy_prediction",
        "delayed_reporting_prediction",
        "under_vaccinated_afp_prediction",
        "suspected_poliovirus_prediction",
        "afp_cases",
        "under_vaccinated_afp_cases",
        "adequate_stool_rate",
        "timely_notification_rate",
        "timely_investigation_rate",
        "timely_lab_result_rate",
        "suspected_poliovirus_lab_result_count",
        "afp_surveillance_quality_score",
        "polio_surveillance_risk_score",
    ]
    existing_prediction_cols = [column for column in prediction_cols if column in predictions.columns]

    output_paths = {
        "training_matrix": str(safe_to_csv(df, args.training_matrix)),
        "predictions": str(safe_to_csv(predictions[existing_prediction_cols], args.output_dir / "polio_afp_next_month_predictions.csv")),
        "alerts": str(safe_to_csv(latest_alerts, args.output_dir / "polio_afp_next_month_surveillance_alerts.csv")),
        "zone_watch": str(safe_to_csv(zone_watch, args.output_dir / "polio_afp_zone_watch_alerts.csv")),
        "signal_trends": str(safe_to_csv(signal_trends, args.output_dir / "polio_afp_signal_trends.csv")),
        "recommendations": str(safe_to_csv(recommendations, args.output_dir / "polio_afp_preparedness_recommendations.csv")),
        "model_comparison": str(safe_to_csv(comparison, args.output_dir / "polio_surveillance_model_comparison.csv")),
        "feature_importance": str(safe_to_csv(feature_importance, args.output_dir / "polio_surveillance_feature_importance.csv")),
        "confusion_matrix": str(safe_to_csv(confusion, args.output_dir / "polio_surveillance_confusion_matrix.csv")),
    }

    metrics = {
        "model_goal": "Predict next-month AFP surveillance-risk and preparedness alert signals.",
        "input_file": str(args.input),
        "training_matrix": str(args.training_matrix),
        "output_dir": str(args.output_dir),
        "split_policy": {
            "train_calibration_prediction_years": "2021-2023",
            "validation_prediction_year": args.validation_year,
            "test_demo_prediction_year": args.test_year,
        },
        "rows": {
            "input_rows": int(len(df)),
            "labelled_rows_all_targets": int(df[TARGET_COLUMNS].notna().all(axis=1).sum()),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "latest_prediction_rows": int((predictions["prediction_period_start"] == predictions["prediction_period_start"].max()).sum()),
        },
        "target_positive_rates": {
            TARGET_DISPLAY[target]: {
                "train": positive_rate(train, target),
                "validation": positive_rate(validation, target),
                "test": positive_rate(test, target),
            }
            for target in TARGET_COLUMNS
        },
        "thresholds": thresholds,
        "target_metrics": {
            TARGET_DISPLAY[result.target_column]: {
                "target_column": result.target_column,
                "validation": result.validation_metrics,
                "test_2025": result.test_metrics,
            }
            for result in results
        },
        "outputs": {
            "latest_alert_rows": int(len(latest_alerts)),
            "zone_watch_rows": int(len(zone_watch)),
            "recommendation_rows": int(len(recommendations)),
            "paths": output_paths,
        },
        "limitation": "Derived AFP surveillance-risk model; not a confirmed polio outbreak prediction model.",
    }
    artifact_path = save_artifact(args.output_dir / "polio_surveillance_model_artifact.joblib", results, metrics)
    metrics["outputs"]["paths"]["artifact"] = str(artifact_path)
    metrics_path = args.output_dir / "polio_surveillance_evaluation_metrics.json"
    metrics["outputs"]["paths"]["metrics"] = str(metrics_path)
    metrics["outputs"]["paths"]["report"] = str(args.report)
    metrics_path.write_text(json.dumps(json_safe(metrics), indent=2), encoding="utf-8")
    write_report(args.report, metrics, comparison, feature_importance, latest_alerts)
    print(json.dumps(json_safe({
        "targets": [TARGET_DISPLAY[target] for target in TARGET_COLUMNS],
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "latest_alert_rows": len(latest_alerts),
        "zone_watch_rows": len(zone_watch),
        "outputs": metrics["outputs"]["paths"],
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
