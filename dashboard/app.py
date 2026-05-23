"""Streamlit dashboard for vaccine-preventable disease decision support."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MEASLES_OUTPUT_DIR = ROOT / "model_outputs_xgboost"
POLIO_OUTPUT_DIR = ROOT / "model_outputs_polio_afp"


@st.cache_data
def load_csv(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_json(path_text: str) -> dict:
    path = Path(path_text)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_measles_csv(name: str) -> pd.DataFrame:
    return load_csv(str(MEASLES_OUTPUT_DIR / name))


def load_polio_csv(name: str) -> pd.DataFrame:
    return load_csv(str(POLIO_OUTPUT_DIR / name))


def require_measles_outputs() -> bool:
    required = [
        "model_comparison.csv",
        "confusion_matrix.csv",
        "feature_importance.csv",
        "top_risk_woredas_latest.csv",
        "next_month_alert_woredas.csv",
        "new_2025_outbreak_locations.csv",
    ]
    missing = [name for name in required if not (MEASLES_OUTPUT_DIR / name).exists()]
    if missing:
        st.error("Missing measles dashboard outputs.")
        st.code("python scripts\\train_measles_mvp_model.py --primary-model xgboost --output-dir model_outputs_xgboost")
        st.write("Missing files:", ", ".join(missing))
        return False
    return True


def require_polio_outputs() -> bool:
    required = [
        "polio_afp_next_month_predictions.csv",
        "polio_afp_next_month_surveillance_alerts.csv",
        "polio_afp_zone_watch_alerts.csv",
        "polio_afp_signal_trends.csv",
        "polio_afp_preparedness_recommendations.csv",
        "polio_surveillance_model_comparison.csv",
        "polio_surveillance_feature_importance.csv",
        "polio_surveillance_confusion_matrix.csv",
        "polio_surveillance_evaluation_metrics.json",
    ]
    missing = [name for name in required if not (POLIO_OUTPUT_DIR / name).exists()]
    if missing:
        st.info("Missing polio surveillance-risk outputs.")
        st.code("python scripts\\train_polio_surveillance_risk_model.py")
        st.write("Missing files:", ", ".join(missing))
        return False
    return True


def render_measles_dashboard() -> None:
    st.subheader("Measles Outbreak Risk")
    st.caption("Next-month outbreak-risk prediction for vaccination outreach and preparedness.")

    if not require_measles_outputs():
        return

    comparison = load_measles_csv("model_comparison.csv")
    confusion = load_measles_csv("confusion_matrix.csv")
    importance = load_measles_csv("feature_importance.csv")
    top_risk = load_measles_csv("top_risk_woredas_latest.csv")
    alerts = load_measles_csv("next_month_alert_woredas.csv")
    new_2025 = load_measles_csv("new_2025_outbreak_locations.csv")

    latest_period = top_risk["prediction_period_start"].iloc[0] if not top_risk.empty else "unknown"
    xgb_rows = comparison[comparison["model_name"].eq("xgboost")]
    xgb = xgb_rows.iloc[0] if not xgb_rows.empty else comparison.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediction period", latest_period)
    c2.metric("XGBoost ROC-AUC", f"{float(xgb['test_roc_auc']):.3f}")
    c3.metric("XGBoost Recall", f"{float(xgb['test_recall']):.3f}")
    c4.metric("Alert rows", f"{len(alerts):,}")

    st.subheader("Next-Month Alerting")
    left, right = st.columns([1.2, 1])
    with left:
        alert_filter = st.multiselect(
            "Measles alert levels",
            ["critical", "high", "watch", "monitor"],
            default=["critical", "high", "watch"],
        )
        filtered_alerts = alerts[alerts["alert_level"].isin(alert_filter)] if alert_filter else alerts
        st.dataframe(
            filtered_alerts[
                [
                    "admin1_region",
                    "admin2_zone",
                    "admin3_woreda",
                    "risk_probability",
                    "risk_bucket",
                    "alert_role",
                    "alert_level",
                    "source_zone_top_source_woreda",
                ]
            ].head(100),
            use_container_width=True,
        )
    with right:
        level_counts = alerts["alert_level"].value_counts().reset_index()
        level_counts.columns = ["alert_level", "count"]
        st.plotly_chart(px.bar(level_counts, x="alert_level", y="count", title="Measles Alert Counts"), use_container_width=True)

    st.subheader("Top High-Risk Woredas")
    top_n = st.slider("Number of top measles woredas", 10, 100, 25, 5)
    st.dataframe(
        top_risk[
            [
                "admin1_region",
                "admin2_zone",
                "admin3_woreda",
                "risk_probability",
                "risk_bucket",
                "risk_prediction",
                "actual_label_available",
            ]
        ].head(top_n),
        use_container_width=True,
    )

    st.subheader("Model Evaluation")
    c1, c2 = st.columns(2)
    with c1:
        metrics = comparison.melt(
            id_vars=["model_name", "model_role"],
            value_vars=["test_precision", "test_recall", "test_f2", "test_roc_auc"],
            var_name="metric",
            value_name="value",
        )
        st.plotly_chart(px.bar(metrics, x="model_name", y="value", color="metric", barmode="group"), use_container_width=True)
    with c2:
        z = confusion[["predicted_0", "predicted_1"]].to_numpy()
        fig = go.Figure(data=go.Heatmap(z=z, x=["Predicted 0", "Predicted 1"], y=["Actual 0", "Actual 1"], text=z, texttemplate="%{text}"))
        fig.update_layout(title="XGBoost Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance")
    top_features = importance.head(20).copy()
    st.plotly_chart(
        px.bar(
            top_features.sort_values("importance_share"),
            x="importance_share",
            y="feature",
            orientation="h",
            color="feature_family",
            title="Top XGBoost Drivers",
        ),
        use_container_width=True,
    )

    st.subheader("New 2025 Outbreak Locations")
    st.write("Woredas whose first recorded outbreak month occurred in 2025.")
    st.dataframe(new_2025.head(100), use_container_width=True)


def render_polio_dashboard() -> None:
    st.subheader("Polio AFP Surveillance Risk")
    st.caption("Next-month AFP surveillance-risk and preparedness alerts. These are not confirmed polio outbreak probabilities.")

    if not require_polio_outputs():
        return

    predictions = load_polio_csv("polio_afp_next_month_predictions.csv")
    alerts = load_polio_csv("polio_afp_next_month_surveillance_alerts.csv")
    zone_watch = load_polio_csv("polio_afp_zone_watch_alerts.csv")
    trends = load_polio_csv("polio_afp_signal_trends.csv")
    recommendations = load_polio_csv("polio_afp_preparedness_recommendations.csv")
    comparison = load_polio_csv("polio_surveillance_model_comparison.csv")
    importance = load_polio_csv("polio_surveillance_feature_importance.csv")
    confusion = load_polio_csv("polio_surveillance_confusion_matrix.csv")
    metrics = load_json(str(POLIO_OUTPUT_DIR / "polio_surveillance_evaluation_metrics.json"))

    latest_period = alerts["prediction_period_start"].iloc[0] if not alerts.empty else "unknown"
    critical_count = int(alerts["alert_level"].eq("critical").sum()) if "alert_level" in alerts else 0
    high_count = int(alerts["alert_level"].eq("high").sum()) if "alert_level" in alerts else 0
    test_rows = metrics.get("rows", {}).get("test_rows", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediction period", latest_period)
    c2.metric("Alert rows", f"{len(alerts):,}")
    c3.metric("Critical / high", f"{critical_count:,} / {high_count:,}")
    c4.metric("2025 test rows", f"{test_rows:,}")

    st.subheader("Latest Surveillance Alerts")
    left, right = st.columns([1.35, 1])
    with left:
        level_options = ["critical", "high", "watch", "monitor"]
        selected_levels = st.multiselect("Polio alert levels", level_options, default=["critical", "high", "watch"])
        filtered_alerts = alerts[alerts["alert_level"].isin(selected_levels)] if selected_levels else alerts
        st.dataframe(
            filtered_alerts[
                [
                    "admin1_region",
                    "admin2_zone",
                    "admin3_woreda",
                    "high_surveillance_risk_probability",
                    "poor_stool_adequacy_probability",
                    "delayed_reporting_probability",
                    "under_vaccinated_afp_probability",
                    "suspected_poliovirus_probability",
                    "alert_level",
                    "alert_reasons",
                    "recommended_action",
                ]
            ].head(100),
            use_container_width=True,
        )
    with right:
        alert_counts = alerts["alert_level"].value_counts().reset_index()
        alert_counts.columns = ["alert_level", "count"]
        st.plotly_chart(
            px.bar(
                alert_counts,
                x="alert_level",
                y="count",
                color="alert_level",
                title="Polio Surveillance Alert Counts",
                category_orders={"alert_level": ["critical", "high", "watch", "monitor"]},
            ),
            use_container_width=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.scatter(
                alerts,
                x="high_surveillance_risk_probability",
                y="suspected_poliovirus_probability",
                size="under_vaccinated_afp_probability",
                color="alert_level",
                hover_name="admin3_woreda",
                hover_data=["admin1_region", "admin2_zone"],
                title="Alert Probability Profile",
            ),
            use_container_width=True,
        )
    with c2:
        if zone_watch.empty:
            st.info("No same-zone watch alerts in the latest polio output.")
        else:
            st.dataframe(
                zone_watch[
                    [
                        "admin1_region",
                        "admin2_zone",
                        "admin3_woreda",
                        "source_zone_top_source_woreda",
                        "alert_reasons",
                    ]
                ].head(50),
                use_container_width=True,
            )

    st.subheader("Preparedness Recommendations")
    st.dataframe(
        recommendations[
            [
                "admin1_region",
                "admin2_zone",
                "admin3_woreda",
                "alert_level",
                "alert_reasons",
                "recommended_action",
            ]
        ].head(100),
        use_container_width=True,
    )

    st.subheader("Surveillance Signal Trends")
    if not trends.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.line(
                    trends,
                    x="period_start",
                    y=[
                        "afp_cases",
                        "under_vaccinated_afp_rows",
                        "suspected_poliovirus_signal_rows",
                    ],
                    title="Monthly AFP Signal Counts",
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                px.line(
                    trends,
                    x="period_start",
                    y=["mean_afp_surveillance_quality_score", "mean_polio_surveillance_risk_score"],
                    title="Monthly Quality and Risk Scores",
                ),
                use_container_width=True,
            )

    st.subheader("Polio Model Evaluation")
    c1, c2 = st.columns(2)
    with c1:
        metric_frame = comparison.melt(
            id_vars=["target_name"],
            value_vars=["test_precision", "test_recall", "test_f2", "test_pr_auc"],
            var_name="metric",
            value_name="value",
        )
        st.plotly_chart(px.bar(metric_frame, x="target_name", y="value", color="metric", barmode="group"), use_container_width=True)
    with c2:
        st.dataframe(
            comparison[
                [
                    "target_name",
                    "threshold",
                    "test_positive_rows",
                    "test_precision",
                    "test_recall",
                    "test_f2",
                    "test_pr_auc",
                ]
            ],
            use_container_width=True,
        )

    selected_target = st.selectbox("Feature importance target", sorted(importance["target_name"].dropna().unique().tolist()))
    top_features = importance[importance["target_name"].eq(selected_target)].head(20)
    st.plotly_chart(
        px.bar(
            top_features.sort_values("importance_share"),
            x="importance_share",
            y="feature",
            orientation="h",
            title=f"Top Polio Drivers: {selected_target}",
        ),
        use_container_width=True,
    )

    st.subheader("Confusion Matrix Summary")
    st.dataframe(confusion, use_container_width=True)

    st.info(
        "Polio outputs are derived AFP surveillance-risk alerts. They should support surveillance review, "
        "immunization outreach, and preparedness planning, not replace official confirmation or outbreak declaration."
    )


def main() -> None:
    st.set_page_config(page_title="VPD Risk Decision Support", layout="wide")
    st.title("Vaccine-Preventable Disease Risk Decision Support")
    measles_tab, polio_tab = st.tabs(["Measles outbreak risk", "Polio AFP surveillance risk"])
    with measles_tab:
        render_measles_dashboard()
    with polio_tab:
        render_polio_dashboard()


if __name__ == "__main__":
    main()
