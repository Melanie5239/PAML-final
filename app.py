import contextlib
import io
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import MODEL1_FINAL_前端 as ml


st.set_page_config(
    page_title="Bank Marketing ML Dashboard",
    page_icon="📊",
    layout="wide",
)


MENU = {
    "Data Exploration": "get_dataset_summary",
    "Feature Description": "get_feature_descriptions",
    "Preprocessing": "run_preprocessing_demo",
    "Training": "train_pipeline",
    "Evaluation": "run_evaluation",
    "Threshold Analysis": "run_threshold_analysis",
    "Feature Analysis": "run_feature_analysis",
    "Prediction": "predict_single",
}


LOCAL_DATA = Path(__file__).with_name("bank-additional-full.csv")
CACHE_VERSION = "prediction-nan-fix-v2"


st.markdown(
    """
    <style>
    .stApp {
        background: #f8fafc;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 2.4rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: #f1f5f9;
        border-right: 1px solid #e2e8f0;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] > label {
        font-size: 1.05rem;
        font-weight: 700;
        color: #334155;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 0.7rem 0.85rem;
        border-radius: 8px;
        margin-bottom: 0.2rem;
        background: transparent;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: #e2e8f0;
    }

    h1, h2, h3 {
        color: #1f2937;
        letter-spacing: 0;
    }

    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
    }

    .section-note {
        color: #64748b;
        font-size: 1.02rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_local_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(LOCAL_DATA, sep=";")
    X = data.drop(columns=["y"])
    y = data[["y"]]
    return X, y


@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if LOCAL_DATA.exists():
        return _read_local_data()
    with contextlib.redirect_stdout(io.StringIO()):
        return ml.load_data()


@st.cache_data(show_spinner=False)
def split_data(X: pd.DataFrame, y: pd.DataFrame):
    return ml.train_test_split_df(X, y, test_size=0.2, random_seed=42)


@st.cache_data(show_spinner=False)
def dataset_summary(X: pd.DataFrame, y: pd.DataFrame) -> dict:
    return ml.get_dataset_summary(X, y)


@st.cache_data(show_spinner=False)
def feature_descriptions() -> pd.DataFrame:
    return ml.get_feature_descriptions()


@st.cache_data(show_spinner="Preparing preprocessing demo...")
def preprocessing_demo(X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame) -> dict:
    return ml.run_preprocessing_demo(X_train_raw, X_test_raw)


@st.cache_resource(show_spinner="Training model once for this session...")
def trained_pipeline(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train_raw: pd.DataFrame,
    y_test_raw: pd.DataFrame,
    cache_version: str,
) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        return ml.train_pipeline(
            X_train_raw,
            X_test_raw,
            y_train_raw,
            y_test_raw,
            learning_rate=0.01,
            n_iterations=1000,
        )


@st.cache_data(show_spinner=False)
def evaluation_result(pipeline_signature: str, threshold: float) -> dict:
    state = st.session_state[pipeline_signature]
    return ml.run_evaluation(
        state["model"],
        state["X_train"],
        state["y_train"],
        state["X_test"],
        state["y_test"],
        threshold=threshold,
    )


@st.cache_data(show_spinner="Running threshold analysis...")
def threshold_result(pipeline_signature: str) -> dict:
    state = st.session_state[pipeline_signature]
    return ml.run_threshold_analysis(state["model"], state["X_test"], state["y_test"])


@st.cache_data(show_spinner="Running feature analysis...")
def feature_result(pipeline_signature: str, top_n: int) -> dict:
    state = st.session_state[pipeline_signature]
    with contextlib.redirect_stdout(io.StringIO()):
        return ml.run_feature_analysis(
            state["model"],
            state["preprocessor"],
            state["X_train_raw"],
            state["X_test_raw"],
            state["y_train"],
            state["y_test"],
            top_n=top_n,
        )


def metric_grid(metrics: dict, columns: int = 4):
    cols = st.columns(columns)
    for idx, (label, value) in enumerate(metrics.items()):
        display = f"{value:.4f}" if isinstance(value, float) else value
        cols[idx % columns].metric(label, display)


def show_table(df: pd.DataFrame, height: int | None = None):
    if height is None:
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.dataframe(df, width="stretch", hide_index=True, height=height)


def show_dict_table(data: dict, key_name: str = "Item", value_name: str = "Value"):
    if not data:
        st.info("No values to display.")
        return
    show_table(pd.DataFrame([{key_name: k, value_name: v} for k, v in data.items()]))


def plot_loss(loss_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_df["iteration"], loss_df["loss"], color="#2563eb", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(alpha=0.25)
    st.pyplot(fig, width="stretch")


def plot_coefficients(df: pd.DataFrame, title: str, color: str):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ordered = df.sort_values("coefficient")
    ax.barh(ordered["feature"], ordered["coefficient"], color=color)
    ax.set_title(title)
    ax.set_xlabel("Coefficient")
    ax.grid(axis="x", alpha=0.25)
    st.pyplot(fig, width="stretch")


def plot_probability_donut(prob: float):
    if not np.isfinite(prob):
        prob = 0.0
    prob = min(max(float(prob), 0.0), 1.0)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    values = [prob, 1 - prob]
    colors = ["#2563eb", "#dbeafe"]

    ax.pie(
        values,
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops=dict(width=0.28, edgecolor="white"),
    )
    ax.text(
        0,
        0.02,
        f"{prob:.0%}",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#1e3a8a",
    )
    ax.text(
        0,
        -0.18,
        "subscription likelihood",
        ha="center",
        va="center",
        fontsize=9,
        color="#64748b",
    )
    ax.set(aspect="equal")
    plt.tight_layout()
    return fig


def complete_prediction_features(visible_features: dict, X_reference: pd.DataFrame) -> dict:
    model_features = {}
    for column in X_reference.columns:
        if column in visible_features:
            model_features[column] = visible_features[column]
        elif pd.api.types.is_numeric_dtype(X_reference[column]):
            model_features[column] = float(X_reference[column].median())
        else:
            mode = X_reference[column].mode()
            model_features[column] = mode.iloc[0] if not mode.empty else "unknown"
    model_features.update(visible_features)
    return model_features


def prediction_form(pipeline_state: dict, X_reference: pd.DataFrame):
    st.markdown(
        '<p class="section-note">Enter customer information, submit, and score it with `predict_single` using the cached trained model.</p>',
        unsafe_allow_html=True,
    )
    defaults = {
        "age": 35,
        "job": "admin.",
        "marital": "single",
        "education": "primary",
        "balance": 1000,
        "housing": "yes",
        "loan": "yes",
        "contact": "cellular",
        "campaign": 1,
        "poutcome": "success",
    }

    form_col, result_col = st.columns([1.35, 1], gap="large")

    with form_col:
        st.subheader("Customer Information")
        if st.button("Reset Inputs"):
            for key, value in defaults.items():
                st.session_state[f"pred_{key}"] = value

        for key, value in defaults.items():
            st.session_state.setdefault(f"pred_{key}", value)

        job_options = [
            "admin.",
            "blue-collar",
            "entrepreneur",
            "housemaid",
            "management",
            "retired",
            "self-employed",
            "services",
            "student",
            "technician",
            "unemployed",
            "unknown",
        ]
        education_options = [
            "primary",
            "secondary",
            "tertiary",
            "basic.4y",
            "basic.6y",
            "basic.9y",
            "high.school",
            "professional.course",
            "university.degree",
            "unknown",
        ]

        with st.form("prediction_form"):
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                age = st.slider("Age", 18, 100, st.session_state.pred_age, key="pred_age")
                job = st.selectbox(
                    "Job",
                    job_options,
                    index=job_options.index(st.session_state.pred_job)
                    if st.session_state.pred_job in job_options else 0,
                    key="pred_job",
                )
                marital = st.selectbox(
                    "Marital",
                    ["single", "married", "divorced", "unknown"],
                    key="pred_marital",
                )
                education = st.selectbox(
                    "Education",
                    education_options,
                    index=education_options.index(st.session_state.pred_education)
                    if st.session_state.pred_education in education_options else 0,
                    key="pred_education",
                )
                balance = st.number_input("Balance", value=st.session_state.pred_balance, key="pred_balance")

            with c2:
                housing = st.radio("Housing Loan", ["yes", "no", "unknown"], key="pred_housing")
                loan = st.radio("Personal Loan", ["yes", "no", "unknown"], key="pred_loan")
                contact = st.selectbox(
                    "Contact Type",
                    ["cellular", "telephone", "unknown"],
                    key="pred_contact",
                )
                campaign = st.number_input(
                    "Campaign Contacts",
                    min_value=1,
                    value=st.session_state.pred_campaign,
                    key="pred_campaign",
                )
                poutcome = st.selectbox(
                    "Previous Outcome",
                    ["success", "failure", "nonexistent", "unknown"],
                    key="pred_poutcome",
                )

            submitted = st.form_submit_button("Predict", width="stretch")

    with result_col:
        st.subheader("Prediction Result")
        if not submitted:
            st.info("Enter customer information and click Predict to generate the result.")
            return

        features = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "campaign": campaign,
            "poutcome": poutcome,
        }

        model_features = complete_prediction_features(features, X_reference)
        result = ml.predict_single(
            model_features,
            model=pipeline_state["model"],
            preprocessor=pipeline_state["preprocessor"],
        )
        probability = float(result["probability"])
        primary_label = "Subscribe" if result["predictions"].get(0.5, 0) else "Not Subscribe"

        m1, m2 = st.columns(2)
        m1.metric("Predicted Outcome", primary_label)
        m2.metric("Probability", result["probability_pct"])
        st.metric("Confidence", result["risk_level"])

        d1, d2 = st.columns([1, 1.1], gap="small")
        with d1:
            st.pyplot(plot_probability_donut(probability), width="content")
        with d2:
            st.markdown("**Recommended Action**")
            st.success(result["interpretation"])

        st.subheader("Threshold Decisions")
        show_table(result["predictions_df"])


X_raw, y_raw = load_data()
X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_data(X_raw, y_raw)
pipeline_state = trained_pipeline(X_train_raw, X_test_raw, y_train_raw, y_test_raw, CACHE_VERSION)
st.session_state["cached_pipeline_state"] = pipeline_state

with st.sidebar:
    st.markdown("### Menu")
    selected = st.radio(
        "Pipeline step",
        list(MENU.keys()),
        index=0,
        label_visibility="collapsed",
    )

st.title("Bank Marketing Subscription Prediction")
st.caption(f"Selected API: `{MENU[selected]}`")


if selected == "Data Exploration":
    summary = dataset_summary(X_raw, y_raw)
    metric_grid(
        {
            "Total Samples": f"{summary['total_samples']:,}",
            "Features": summary["n_features"],
            "Subscribed": f"{summary['n_subscribed']:,}",
            "Positive Rate": f"{summary['positive_rate_pct']:.2f}%",
        }
    )
    st.subheader("Class Distribution")
    st.bar_chart(summary["class_distribution_df"].set_index("label")["count"])
    st.subheader("Numeric Feature Statistics")
    show_table(summary["numeric_stats_df"])
    st.subheader("Unknown Value Audit")
    show_dict_table(summary["unknown_counts"], "Feature", "Unknown Count")

elif selected == "Feature Description":
    st.markdown('<p class="section-note">Feature metadata returned by `get_feature_descriptions`.</p>', unsafe_allow_html=True)
    show_table(feature_descriptions(), height=560)

elif selected == "Preprocessing":
    demo = preprocessing_demo(X_train_raw, X_test_raw)
    metric_grid(
        {
            "Train Samples": f"{demo['n_train_samples']:,}",
            "Test Samples": f"{demo['n_test_samples']:,}",
            "Original Features": demo["n_features_original"],
            "Encoded Features": demo["n_features_encoded"],
        }
    )
    st.subheader("Pipeline Steps")
    st.write(pd.DataFrame({"Step": demo["steps"]}))
    st.subheader("Unknown Counts")
    show_dict_table(demo["unknown_counts"], "Feature", "Unknown Count")
    st.subheader("Imputation Values")
    show_table(demo["imputation_df"])
    st.subheader("One-Hot Encoding Summary")
    show_table(demo["ohe_summary_df"])
    st.subheader("Scaling Check")
    show_table(demo["scaling_df"])

elif selected == "Training":
    metric_grid(
        {
            "Learning Rate": pipeline_state["learning_rate"],
            "Iterations": pipeline_state["n_iterations"],
            "Encoded Features": pipeline_state["n_features"],
            "Final Loss": pipeline_state["final_loss"],
        }
    )
    st.subheader("Test Metrics")
    metric_grid(pipeline_state["test_metrics"])
    st.subheader("Training Loss")
    plot_loss(pipeline_state["loss_df"])
    st.subheader("Saved Artifacts")
    show_dict_table(
        {
            "Model": pipeline_state["model_path"],
            "Preprocessor": pipeline_state["preprocessor_path"],
        },
        "Artifact",
        "Path",
    )

elif selected == "Evaluation":
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
    result = evaluation_result("cached_pipeline_state", threshold)
    st.subheader("Metrics")
    show_table(result["metrics_df"])
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Train Confusion Matrix")
        show_dict_table(result["train_cm"], "Cell", "Count")
    with col2:
        st.subheader("Test Confusion Matrix")
        show_dict_table(result["test_cm"], "Cell", "Count")
    st.subheader("Sample Predictions")
    show_table(result["sample_predictions_df"])

elif selected == "Threshold Analysis":
    result = threshold_result("cached_pipeline_state")
    metric_grid(
        {
            "Best F1 Threshold": result["best_f1_threshold"],
            "Best F1": result["best_f1_value"],
            "Best Recall Threshold": result["best_recall_threshold"],
            "Best Recall": result["best_recall_value"],
        }
    )
    st.subheader("Sensitivity")
    st.line_chart(result["sensitivity_df"].set_index("Threshold"))
    show_table(result["sensitivity_df"])
    st.subheader("Business Comparison")
    show_table(result["business_df"])
    st.info(result["interpretation"])

elif selected == "Feature Analysis":
    top_n = st.slider("Top features", 5, 20, 10, 1)
    result = feature_result("cached_pipeline_state", top_n)
    metric_grid(
        {
            "Most Positive": result["most_positive_feature"],
            "Positive Coef": result["most_positive_coef"],
            "Most Negative": result["most_negative_feature"],
            "Negative Coef": result["most_negative_coef"],
        },
        columns=2,
    )
    col1, col2 = st.columns(2)
    with col1:
        plot_coefficients(result["top_positive_df"], "Top Positive Coefficients", "#16a34a")
    with col2:
        plot_coefficients(result["top_negative_df"], "Top Negative Coefficients", "#dc2626")
    st.subheader("Learning Rate Sensitivity")
    show_table(result["lr_sensitivity_df"])
    st.subheader("Duration Ablation")
    show_table(result["ablation_df"])

elif selected == "Prediction":
    prediction_form(pipeline_state, X_raw)
