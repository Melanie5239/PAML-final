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


PREFERRED_DATA = Path(__file__).with_name("bank-full.csv")
FALLBACK_DATA = Path(__file__).with_name("bank-additional-full.csv")
CACHE_VERSION = "local-data-fallback-v5"


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
        font-size: 18px;
    }

    .block-container {
        max-width: 1320px;
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f6f9ff 0%, #ebf2ff 100%);
        border-right: 1px solid rgba(59,130,246,0.10);
    }

    section[data-testid="stSidebar"] * {
        color: #1e3a8a !important;
    }

    section[data-testid="stSidebar"] h3 {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.7rem !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 1.38rem !important;
        font-weight: 800 !important;
        line-height: 1.45 !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 0.85rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.28rem;
        background: transparent;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.80);
    }

    h1, h2, h3 {
        color: #1f2937;
        letter-spacing: 0;
    }

    h1 {
        font-size: 3.25rem !important;
        font-weight: 850 !important;
        line-height: 1.15 !important;
        color: #1e3a8a !important;
    }

    h2 {
        font-size: 2.35rem !important;
        font-weight: 820 !important;
        color: #1e3a8a !important;
    }

    h3 {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #1d4ed8 !important;
    }

    .stCaptionContainer,
    .stCaptionContainer p {
        font-size: 1.2rem !important;
        color: #64748b !important;
    }

    .stMarkdown p,
    .stInfo p,
    .stSuccess p {
        font-size: 1.24rem !important;
        line-height: 1.6 !important;
    }

    .stSelectbox label,
    .stNumberInput label,
    .stSlider label,
    .stRadio label,
    .stTextInput label {
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        color: #1e3a8a !important;
    }

    div[data-baseweb="select"] > div {
        min-height: 62px !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
    }

    div[data-baseweb="select"] span,
    div[role="listbox"] ul li,
    div[role="option"] {
        font-size: 1.28rem !important;
        font-weight: 700 !important;
    }

    .stNumberInput input {
        min-height: 60px !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
    }

    div[role="radiogroup"] label p,
    div[role="radiogroup"] label span {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
    }

    .stSlider div[data-baseweb="slider"] * {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        min-height: 3.6rem !important;
        font-size: 1.28rem !important;
        font-weight: 800 !important;
        border-radius: 12px !important;
        background: linear-gradient(90deg, #3b82f6, #2563eb) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 8px 18px rgba(37,99,235,0.18) !important;
    }

    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid rgba(59,130,246,0.12);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px rgba(37,99,235,0.07);
    }

    div[data-testid="metric-container"] label {
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        color: #1e3a8a !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.95rem !important;
        font-weight: 850 !important;
        color: #0f172a !important;
    }

    .section-note {
        color: #64748b;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stDataFrame"] {
        font-size: 1.2rem !important;
    }

    div[data-testid="stDataFrame"] * {
        font-size: 1.12rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_local_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = PREFERRED_DATA if PREFERRED_DATA.exists() else FALLBACK_DATA
    data = pd.read_csv(data_path, sep=";")
    X = data.drop(columns=["y"])
    y = data[["y"]]
    return X, y


@st.cache_data(show_spinner="Loading dataset...")
def load_data(data_version: str = "bank-full-v1") -> tuple[pd.DataFrame, pd.DataFrame]:
    """data_version forces cache invalidation when the source file changes."""
    if PREFERRED_DATA.exists() or FALLBACK_DATA.exists():
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


def trained_pipeline(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train_raw: pd.DataFrame,
    y_test_raw: pd.DataFrame,
    cache_version: str,
) -> dict:
    """Train fresh each session — no persistent cache to avoid stale preprocessing."""
    if "pipeline_result" not in st.session_state:
        with st.spinner("Training model (lr=0.1, 3000 iterations)..."):
            with contextlib.redirect_stdout(io.StringIO()):
                st.session_state["pipeline_result"] = ml.train_pipeline(
                    X_train_raw,
                    X_test_raw,
                    y_train_raw,
                    y_test_raw,
                    learning_rate=0.1,
                    n_iterations=3000,
                )
    return st.session_state["pipeline_result"]


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


def threshold_result(pipeline_signature: str) -> dict:
    state = st.session_state[pipeline_signature]
    return ml.run_threshold_analysis(state["model"], state["X_test"], state["y_test"])


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


X_raw, y_raw = load_data("bank-full-v1")
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
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.15, 0.05)
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
    st.markdown("Threshold sensitivity evaluated on the held-out test set.")
    state = st.session_state["cached_pipeline_state"]
    thresholds = [round(t * 0.05, 2) for t in range(2, 20)]
    rows = []
    for t in thresholds:
        y_pred = state["model"].predict(state["X_test"], threshold=t)
        m = ml.evaluate(state["y_test"], y_pred)
        rows.append({"Threshold": t,
                     "Accuracy":  round(m["Accuracy"],  4),
                     "Precision": round(m["Precision"], 4),
                     "Recall":    round(m["Recall"],    4),
                     "F1-Score":  round(m["F1-Score"],  4)})
    sens_df = pd.DataFrame(rows)
    best_f1  = sens_df.loc[sens_df["F1-Score"].idxmax()]
    best_rec = sens_df.loc[sens_df["Recall"].idxmax()]
    metric_grid({
        "Best F1 Threshold":     float(best_f1["Threshold"]),
        "Best F1":               float(best_f1["F1-Score"]),
        "Best Recall Threshold": float(best_rec["Threshold"]),
        "Best Recall":           float(best_rec["Recall"]),
    })
    st.subheader("Sensitivity Chart")
    st.line_chart(sens_df.set_index("Threshold")[["Accuracy", "Precision", "Recall", "F1-Score"]])
    st.subheader("Sensitivity Table")
    show_table(sens_df)
    st.info(
        f"Best F1={float(best_f1['F1-Score']):.4f} at threshold {float(best_f1['Threshold'])}. "
        "Lowering the threshold increases Recall (fewer missed subscribers) "
        "at the cost of lower Precision (more unnecessary calls)."
    )

elif selected == "Feature Analysis":
    top_n = st.slider("Top features", 5, 20, 10, 1)
    state = st.session_state["cached_pipeline_state"]
    feature_names = state["model"].weights.__class__.__name__  # just to access
    weights = state["model"].weights
    names   = state["preprocessor"].get_feature_names()
    coef_df = pd.DataFrame({"feature": names, "coefficient": weights})
    coef_df = coef_df.reindex(coef_df["coefficient"].abs().sort_values(ascending=False).index)
    top_pos = coef_df[coef_df["coefficient"] > 0].head(top_n).copy()
    top_neg = coef_df[coef_df["coefficient"] < 0].head(top_n).copy()
    metric_grid({
        "Most Positive Feature": coef_df[coef_df["coefficient"] > 0].iloc[0]["feature"],
        "Positive Coef":         round(float(coef_df[coef_df["coefficient"] > 0].iloc[0]["coefficient"]), 4),
        "Most Negative Feature": coef_df[coef_df["coefficient"] < 0].iloc[0]["feature"],
        "Negative Coef":         round(float(coef_df[coef_df["coefficient"] < 0].iloc[0]["coefficient"]), 4),
    }, columns=2)
    col1, col2 = st.columns(2)
    with col1:
        plot_coefficients(top_pos, "Top Positive Coefficients", "#16a34a")
    with col2:
        plot_coefficients(top_neg, "Top Negative Coefficients", "#dc2626")

elif selected == "Prediction":
    prediction_form(pipeline_state, X_raw)
