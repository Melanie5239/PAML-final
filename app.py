import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Bank Marketing Subscription Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Session state helpers
# -----------------------------
def init_state():
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
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_inputs():
    st.session_state.age = 35
    st.session_state.job = "admin."
    st.session_state.marital = "single"
    st.session_state.education = "primary"
    st.session_state.balance = 1000
    st.session_state.housing = "yes"
    st.session_state.loan = "yes"
    st.session_state.contact = "cellular"
    st.session_state.campaign = 1
    st.session_state.poutcome = "success"


init_state()

# -----------------------------
# Placeholder prediction logic
# -----------------------------
def placeholder_predict(features):
    score = 0.35

    if 30 <= features["age"] <= 55:
        score += 0.08
    if features["balance"] > 1500:
        score += 0.18
    if features["poutcome"] == "success":
        score += 0.22
    if features["contact"] == "cellular":
        score += 0.07
    if features["campaign"] <= 2:
        score += 0.05
    if features["housing"] == "no":
        score += 0.03
    if features["loan"] == "no":
        score += 0.02

    score = min(max(score, 0.05), 0.95)
    label = "Subscribe" if score >= 0.50 else "Not Subscribe"
    return label, score


def get_confidence(prob):
    if prob >= 0.80:
        return "High"
    elif prob >= 0.60:
        return "Medium"
    return "Low"


def get_recommendation(prob):
    if prob >= 0.75:
        return "Prioritize this customer for immediate follow-up."
    elif prob >= 0.55:
        return "Consider follow-up in the next outreach round."
    return "Low priority for the current campaign."


def plot_probability_donut(prob):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    values = [prob, 1 - prob]
    colors = ["#2563eb", "#dbeafe"]

    ax.pie(
        values,
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops=dict(width=0.28, edgecolor="white")
    )

    ax.text(
        0, 0.02,
        f"{prob:.0%}",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#1e3a8a"
    )
    ax.text(
        0, -0.18,
        "subscription likelihood",
        ha="center",
        va="center",
        fontsize=9,
        color="#64748b"
    )

    ax.set(aspect="equal")
    plt.tight_layout()
    return fig


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
/* Global */
.stApp {
    background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
}

.block-container {
    max-width: 1320px;
    padding-top: 2.2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f6f9ff 0%, #ebf2ff 100%);
    border-right: 1px solid rgba(59,130,246,0.10);
    width: 250px !important;
    min-width: 250px !important;
    max-width: 250px !important;
}

section[data-testid="stSidebar"] * {
    color: #1e3a8a !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label {
    font-size: 1.08rem !important;
    line-height: 1.55 !important;
}

.sidebar-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #1d4ed8;
    margin-bottom: 0.6rem;
}

.sidebar-box {
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(59,130,246,0.08);
    border-radius: 16px;
    padding: 1rem;
    box-shadow: 0 8px 22px rgba(37,99,235,0.06);
}

/* Header */
.main-title {
    font-size: 2.65rem;
    font-weight: 800;
    color: #1e3a8a;
    line-height: 1.15;
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}

.subtitle-text {
    font-size: 1.05rem;
    color: #64748b;
    margin-bottom: 1.2rem;
}

/* Top title bars */
.top-title-box {
    background: #ffffff;
    border-radius: 18px;
    padding: 0.95rem 1.2rem;
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.07);
    border: 1px solid rgba(59,130,246,0.08);
    margin-bottom: 0.7rem;
    transition: all 0.25s ease;
}

.top-title-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 28px rgba(37, 99, 235, 0.12);
}

.top-title-text {
    font-size: 1.7rem;
    font-weight: 800;
    color: #1d4ed8;
    line-height: 1.2;
    margin: 0;
}

.how-to-box {
    background: #ffffff;
    border-radius: 18px;
    padding: 1rem 1.2rem;
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.07);
    border: 1px solid rgba(59,130,246,0.08);
    margin-bottom: 1rem;
}

.how-to-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #1d4ed8;
    margin-bottom: 0.35rem;
}

.how-to-text {
    font-size: 1rem;
    color: #475569;
    line-height: 1.6;
    margin: 0;
}

/* Panels */
.panel-box {
    background: #ffffff;
    border-radius: 20px;
    padding: 1.35rem;
    box-shadow: 0 8px 24px rgba(37,99,235,0.08);
    border: 1px solid rgba(59,130,246,0.08);
    transition: all 0.25s ease;
}

.panel-box:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 30px rgba(37,99,235,0.14);
}

/* Field labels */
.big-label {
    font-size: 1.5rem;
    font-weight: 800;
    color: #1e3a8a;
    margin-top: 0.35rem;
    margin-bottom: 0.3rem;
    line-height: 1.2;
}

/* Widget labels */
.stSelectbox label,
.stNumberInput label,
.stSlider label,
.stRadio label {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #1e3a8a !important;
}

/* Inputs */
div[data-baseweb="select"] > div {
    min-height: 54px !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #0f172a !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 3px rgba(96,165,250,0.15) !important;
}

div[data-baseweb="select"] span {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #0f172a !important;
}

div[role="listbox"] ul li,
div[role="option"] {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}

.stNumberInput input {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #0f172a !important;
}

.stNumberInput input:hover {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 3px rgba(96,165,250,0.15) !important;
}

div[role="radiogroup"] label p,
div[role="radiogroup"] label span {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #0f172a !important;
}

.stSlider div[data-baseweb="slider"] * {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #3b82f6, #2563eb);
    color: white;
    font-weight: 800;
    font-size: 1.05rem;
    border-radius: 12px;
    height: 3rem;
    border: none;
    transition: all 0.25s ease;
    box-shadow: 0 8px 18px rgba(37,99,235,0.18);
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 24px rgba(37,99,235,0.28);
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #f5f8ff;
    border-radius: 14px;
    padding: 14px;
    transition: all 0.25s ease;
    border: 1px solid transparent;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    border-color: #bfdbfe;
    box-shadow: 0 10px 24px rgba(59,130,246,0.12);
}

div[data-testid="metric-container"] label {
    font-size: 1.08rem !important;
    font-weight: 800 !important;
    color: #1e3a8a !important;
}

div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
}

/* Result labels */
.result-subtitle {
    font-size: 1.08rem;
    font-weight: 800;
    color: #1e3a8a;
    margin-bottom: 0.45rem;
}

/* Recommendation */
.recommendation {
    background: #e0f2fe;
    padding: 14px 16px;
    border-radius: 12px;
    font-weight: 800;
    font-size: 1.15rem;
    color: #0f172a;
    line-height: 1.4;
    transition: all 0.25s ease;
}

.recommendation:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(14,165,233,0.12);
}

.disclaimer {
    margin-top: 10px;
    font-size: 0.95rem;
    color: #64748b;
    line-height: 1.5;
}

.info-row {
    margin-top: 0.4rem;
    margin-bottom: 0.8rem;
}

.result-divider {
    height: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">Bank Term Deposit Subscription Predictor</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle-text">Predict subscription likelihood from customer and campaign inputs.</div>',
    unsafe_allow_html=True
)
st.markdown(
    """
    <div class="how-to-box">
        <div class="how-to-title">How to Use</div>
        <p class="how-to-text">Enter customer information, click Predict, and review the predicted outcome, probability, and recommended next action.</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.button("Reset Inputs", on_click=reset_inputs, use_container_width=False)

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([1.35, 1], gap="medium")

# -----------------------------
# Left column: form
# -----------------------------
with col1:
    st.markdown(
        '<div class="top-title-box"><div class="top-title-text">Customer Information</div></div>',
        unsafe_allow_html=True
    )

    with st.form("prediction_form"):
        form_col1, form_col2 = st.columns(2, gap="medium")

        with form_col1:
            st.markdown('<div class="big-label">Age</div>', unsafe_allow_html=True)
            age = st.slider("", 18, 100, st.session_state.age, key="age", label_visibility="collapsed")

            st.markdown('<div class="big-label">Job</div>', unsafe_allow_html=True)
            job = st.selectbox(
                "",
                ["admin.", "technician", "services", "management", "retired", "student"],
                index=["admin.", "technician", "services", "management", "retired", "student"].index(st.session_state.job)
                if st.session_state.job in ["admin.", "technician", "services", "management", "retired", "student"]
                else 0,
                key="job",
                label_visibility="collapsed"
            )

            st.markdown('<div class="big-label">Marital</div>', unsafe_allow_html=True)
            marital = st.selectbox(
                "",
                ["single", "married", "divorced"],
                index=["single", "married", "divorced"].index(st.session_state.marital),
                key="marital",
                label_visibility="collapsed"
            )

            st.markdown('<div class="big-label">Education</div>', unsafe_allow_html=True)
            education = st.selectbox(
                "",
                ["primary", "secondary", "tertiary"],
                index=["primary", "secondary", "tertiary"].index(st.session_state.education)
                if st.session_state.education in ["primary", "secondary", "tertiary"] else 0,
                key="education",
                label_visibility="collapsed"
            )

            st.markdown('<div class="big-label">Balance</div>', unsafe_allow_html=True)
            balance = st.number_input("", value=st.session_state.balance, key="balance", label_visibility="collapsed")

        with form_col2:
            st.markdown('<div class="big-label">Housing Loan</div>', unsafe_allow_html=True)
            housing = st.radio(
                "",
                ["yes", "no"],
                index=["yes", "no"].index(st.session_state.housing),
                key="housing",
                label_visibility="collapsed"
            )

            st.markdown('<div class="big-label">Personal Loan</div>', unsafe_allow_html=True)
            loan = st.radio(
                "",
                ["yes", "no"],
                index=["yes", "no"].index(st.session_state.loan),
                key="loan",
                label_visibility="collapsed"
            )

            st.markdown('<div class="big-label">Contact Type</div>', unsafe_allow_html=True)
            contact = st.selectbox(
                "",
                ["cellular", "telephone"],
                index=["cellular", "telephone"].index(st.session_state.contact)
                if st.session_state.contact in ["cellular", "telephone"] else 0,
                key="contact",
                label_visibility="collapsed"
            )

            st.markdown('<div class="big-label">Campaign Contacts</div>', unsafe_allow_html=True)
            campaign = st.number_input("", value=st.session_state.campaign, min_value=1, key="campaign", label_visibility="collapsed")

            st.markdown('<div class="big-label">Previous Outcome</div>', unsafe_allow_html=True)
            poutcome = st.selectbox(
                "",
                ["success", "failure", "unknown"],
                index=["success", "failure", "unknown"].index(st.session_state.poutcome)
                if st.session_state.poutcome in ["success", "failure", "unknown"] else 0,
                key="poutcome",
                label_visibility="collapsed"
            )

        st.write("")
        submit = st.form_submit_button("Predict", use_container_width=True)

# -----------------------------
# Right column: results
# -----------------------------
with col2:
    st.markdown(
        '<div class="top-title-box"><div class="top-title-text">Prediction Result</div></div>',
        unsafe_allow_html=True
    )

    if submit:
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

        label, prob = placeholder_predict(features)
        confidence = get_confidence(prob)
        recommendation = get_recommendation(prob)

        metric_col1, metric_col2 = st.columns(2, gap="small")
        with metric_col1:
            st.metric("Predicted Outcome", label)
        with metric_col2:
            st.metric("Probability", f"{prob:.2%}")

        st.markdown('<div class="info-row"></div>', unsafe_allow_html=True)
        st.metric("Confidence", confidence)

        st.markdown('<div class="result-divider"></div>', unsafe_allow_html=True)

        donut_col1, donut_col2 = st.columns([1, 1.1], gap="small")
        with donut_col1:
            donut_fig = plot_probability_donut(prob)
            st.pyplot(donut_fig, use_container_width=False)

        with donut_col2:
            st.markdown('<div class="result-subtitle">Recommended Action</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="recommendation">{recommendation}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="disclaimer">This tool supports decision-making only and should not replace human judgment.</div>',
                unsafe_allow_html=True
            )

    else:
        st.info("Enter customer information and click Predict to generate the result.")
