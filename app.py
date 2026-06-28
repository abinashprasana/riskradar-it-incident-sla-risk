import os
import time
import pathlib
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

load_dotenv()

from data_processing import load_event_log, build_incident_summary
from decision_logic import risk_band, recommended_action
from llm_explainer import explain

_HERE         = pathlib.Path(__file__).parent
DEFAULT_CSV   = str(_HERE / "incident_event_log.csv")
DEFAULT_MODEL = str(_HERE / "best_model.joblib")

st.set_page_config(page_title="RiskRadar", layout="wide", page_icon="🚦")

# ── Colour palette ────────────────────────────────────────────────────────────
C_RED     = "#e63946"   # primary / high risk
C_AMBER   = "#f4a261"   # medium risk
C_GREEN   = "#2a9d8f"   # low risk / good
C_BLUE    = "#5b9ec9"   # informational (lighter for dark bg)
C_NAVY    = "#e2e8f0"   # primary text on dark
C_MUTED   = "#94a3b8"   # muted text on dark
C_BORDER  = "#2d4a7a"   # subtle border on dark

# ── Matplotlib defaults (dark theme) ─────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":   "#1a2744",
    "axes.facecolor":     "#152039",
    "axes.edgecolor":     "#2d4a7a",
    "axes.labelcolor":    "#94a3b8",
    "axes.titlecolor":    "#e2e8f0",
    "axes.grid":          True,
    "grid.alpha":         0.20,
    "grid.color":         "#213255",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelsize":     9.5,
    "axes.titlesize":     10,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "xtick.color":        "#94a3b8",
    "ytick.color":        "#94a3b8",
    "text.color":         "#e2e8f0",
    "legend.facecolor":   "#1a2744",
    "legend.edgecolor":   "#2d4a7a",
    "figure.dpi":         110,
})

# ── CSS overrides ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* Hide Streamlit chrome */
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.2rem !important; }}

/* Page background — deep navy with a subtle blue-purple shift */
[data-testid="stAppViewContainer"] {{
    background: linear-gradient(160deg, #0d1b2a 0%, #0f2240 50%, #0d1b2a 100%) !important;
}}
[data-testid="stAppViewBlockContainer"] {{
    background: transparent !important;
}}

/* Sidebar glass */
[data-testid="stSidebar"] {{
    background: rgba(8, 18, 38, 0.90) !important;
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    box-shadow: 2px 0 24px rgba(0, 0, 0, 0.40);
}}

/* Glass tab bar */
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 14px;
    padding: 5px 6px;
    border: 1px solid rgba(255, 255, 255, 0.09);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.30);
    gap: 3px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px;
    padding: 8px 22px;
    font-weight: 600;
    font-size: 0.875rem;
    color: {C_MUTED};
    transition: all 0.18s ease;
    border: none !important;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: rgba(255, 255, 255, 0.12) !important;
    color: {C_RED} !important;
    box-shadow: 0 2px 14px rgba(0, 0, 0, 0.30) !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{ display: none !important; }}
.stTabs [data-baseweb="tab-border"]    {{ display: none !important; }}

/* Custom KPI cards */
.kpi-card {{
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-top: 3px solid {C_RED};
    border-radius: 12px;
    padding: 18px 20px 16px;
    box-shadow: 0 4px 22px rgba(0, 0, 0, 0.30);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    box-sizing: border-box;
}}
.kpi-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 10px 32px rgba(0, 0, 0, 0.40);
    background: rgba(255, 255, 255, 0.08);
}}
.kpi-label {{
    font-size: 0.68rem;
    color: {C_MUTED};
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 9px;
}}
.kpi-value {{
    font-size: 1.75rem;
    font-weight: 800;
    color: #e2e8f0;
    line-height: 1.1;
}}
.kpi-delta {{
    font-size: 0.73rem;
    font-weight: 600;
    margin-top: 6px;
    opacity: 0.88;
}}

/* Metric cards — Tab 4 */
[data-testid="stMetric"] {{
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-top: 3px solid {C_BLUE};
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.28);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.38);
}}
[data-testid="stMetricLabel"] {{
    color: {C_MUTED} !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
[data-testid="stMetricValue"] {{
    color: #e2e8f0 !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
}}

/* Sidebar section labels */
.sidebar-section {{
    font-size: 0.67rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    padding: 2px 0 7px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    margin-bottom: 10px;
}}

/* Section headings */
h4 {{
    color: #e2e8f0 !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    margin: 1.8rem 0 0.9rem !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.09) !important;
}}

/* Horizontal rule */
hr {{
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.10) 30%, rgba(255,255,255,0.10) 70%, transparent);
    margin: 1.8rem 0;
}}

/* Download button */
div.stDownloadButton > button {{
    background: linear-gradient(135deg, {C_RED} 0%, #c1121f 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 24px;
    width: 100%;
    box-shadow: 0 4px 16px rgba(230, 57, 70, 0.40);
    transition: all 0.2s ease;
}}
div.stDownloadButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 7px 22px rgba(230, 57, 70, 0.55);
}}

/* Sidebar bottom note */
.sidebar-note {{
    font-size: 0.72rem;
    color: #475569;
    text-align: center;
    line-height: 1.7;
    padding-top: 4px;
}}
</style>
""", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, {C_RED} 0%, {C_NAVY} 100%);
    border-radius: 14px;
    padding: 26px 36px;
    margin-bottom: 22px;
">
  <div style="display:flex; align-items:center; gap:16px">
    <span style="font-size:2.6rem; line-height:1">🚦</span>
    <div>
      <h1 style="color:white; margin:0; font-size:2rem; font-weight:800; letter-spacing:-0.02em">
        RiskRadar
      </h1>
      <p style="color:rgba(255,255,255,0.70); margin:5px 0 0; font-size:0.9rem">
        IT Incident SLA Breach Risk &nbsp;·&nbsp; Decision Support Tool
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Cache helpers ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_data
def build_from_path(path: str) -> pd.DataFrame:
    return build_incident_summary(load_event_log(path))


@st.cache_data
def build_from_upload(file) -> pd.DataFrame:
    return build_incident_summary(load_event_log(file))


@st.cache_resource(show_spinner=False)
def train_fresh_model(csv_path: str):
    """Retrain from the CSV when the saved model is incompatible with the runtime."""
    from feature_engineering import make_train_test, build_preprocess_pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    df_events = load_event_log(csv_path)
    df_inc    = build_incident_summary(df_events)
    X_train, _, y_train, _ = make_train_test(df_inc)
    pre  = build_preprocess_pipeline(X_train)
    pipe = Pipeline([
        ("preprocess", pre),
        ("model", RandomForestClassifier(
            n_estimators=150, random_state=42,
            class_weight="balanced_subsample", n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)
    try:
        joblib.dump(pipe, DEFAULT_MODEL, compress=3)
    except Exception:
        pass
    return pipe


def score_incidents(model, df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy()
    drop = [c for c in ["number", "sla_breached", "made_sla", "opened_at_min", "resolved_at_max"]
            if c in df.columns]
    X    = df.drop(columns=drop)
    st.session_state["n_features"] = X.shape[1]
    probs = model.predict_proba(X)[:, 1]
    df["sla_breach_probability"] = probs
    df["risk_band"]          = df["sla_breach_probability"].apply(risk_band)
    df["recommended_action"] = df["risk_band"].apply(recommended_action)
    return df.sort_values("sla_breach_probability", ascending=False).reset_index(drop=True)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_hist(series, bins=20, xlabel=""):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(series, bins=bins, color=C_BLUE, edgecolor="white", alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Incidents")
    plt.tight_layout()
    return fig


def plot_risk_bands(band_counts: pd.Series):
    labels = ["High", "Medium", "Low"]
    values = [band_counts.get(l, 0) for l in labels]
    colors = [C_RED, C_AMBER, C_GREEN]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.55)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.015,
            f"{int(val):,}",
            ha="center", va="bottom", fontsize=9, fontweight="600",
        )
    ax.set_xlabel("Risk band")
    ax.set_ylabel("Incidents")
    ax.set_ylim(0, max(values) * 1.15)
    plt.tight_layout()
    return fig


def plot_donut(labels, values):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    wedge_colors = [C_RED, C_AMBER, C_GREEN][: len(labels)]
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=wedge_colors,
        autopct="%1.0f%%", startangle=90,
        wedgeprops={"width": 0.52, "edgecolor": "white", "linewidth": 2},
        pctdistance=0.78,
    )
    for t in texts:
        t.set_fontsize(9)
    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_color("white")
        at.set_fontweight("bold")
    plt.tight_layout()
    return fig


def plot_heatmap(df: pd.DataFrame):
    if "priority" not in df.columns:
        return None
    pivot = (
        pd.crosstab(df["priority"], df["risk_band"])
        .reindex(columns=["High", "Medium", "Low"]).fillna(0)
    )
    fig, ax = plt.subplots(figsize=(6, 3.8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index], fontsize=8.5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(["High", "Medium", "Low"], fontsize=8.5)
    ax.set_xlabel("Risk band")
    ax.set_ylabel("Priority")
    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cb.ax.tick_params(labelsize=8, colors=C_MUTED)
    cb.set_label("Count", fontsize=8.5, color=C_MUTED)
    mx = float(pivot.values.max()) or 1.0
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = float(pivot.values[i, j])
            ax.text(j, i, f"{int(v):,}", ha="center", va="center",
                    color="white" if v / mx > 0.55 else "#343a40",
                    fontsize=9, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_hbar(labels, values, color, xlabel="Avg risk score"):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y_pos = range(len(labels))
    bars  = ax.barh(list(y_pos), values, color=color, edgecolor="white", alpha=0.88)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    for bar, val in zip(bars, values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color=C_MUTED)
    plt.tight_layout()
    return fig


def plot_calibration(df: pd.DataFrame):
    tmp = df.copy()
    tmp["prob_bin"] = pd.cut(tmp["sla_breach_probability"], bins=10)
    calib = tmp.groupby("prob_bin", observed=False).agg(
        avg_pred=("sla_breach_probability", "mean"),
        actual_rate=("sla_breached", "mean"),
        n=("sla_breached", "count"),
    ).dropna()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(calib["avg_pred"], calib["actual_rate"],
            marker="o", color=C_RED, linewidth=2.2, markersize=7, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color=C_MUTED, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Avg predicted probability (bin)")
    ax.set_ylabel("Actual breach rate (bin)")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    return fig, calib


def plot_confusion(cm, threshold=0.5):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Met", "Breached"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Met", "Breached"])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8, colors=C_MUTED)
    for i in range(2):
        for j in range(2):
            v = int(cm[i, j])
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=12,
                    color="white" if v > cm.max() * 0.6 else "#343a40", fontweight="bold")
    plt.tight_layout()
    return fig


def get_risk_drivers(row: pd.Series, df: pd.DataFrame) -> list[str]:
    drivers = []

    def safe_val(col):
        try:
            v = row[col]
            return None if pd.isna(v) else float(v)
        except Exception:
            return None

    def safe_stat(col, fn):
        try:
            v = fn(df[col])
            return None if pd.isna(v) else float(v)
        except Exception:
            return None

    for col, label, note in [
        ("reassignment_count_max", "Reassignment count", "ticket changed hands too many times"),
        ("reopen_count_max",       "Reopen count",       "fix did not stick the first time"),
        ("resolution_hours",       "Resolution time",    "slow closure pattern"),
        ("total_events",           "Total events",       "lots of activity with no clear closure"),
    ]:
        v   = safe_val(col)
        med = safe_stat(col, lambda s: s.median())
        q75 = safe_stat(col, lambda s: s.quantile(0.75))
        if v is not None and med and q75 and v >= q75 and v > med:
            drivers.append(f"**{label}** is {v:g} ({note})")

    if "priority" in row.index and pd.notna(row["priority"]):
        pr = str(row["priority"])
        if "1" in pr or "Critical" in pr:
            drivers.append("**Priority is Critical.** Needs immediate escalation.")
        elif "2" in pr or "High" in pr:
            drivers.append("**Priority is High.** Worth escalating early.")

    return drivers or ["No major red flags in the usual signals. Looks like a fairly normal ticket."]


# ── Sidebar ───────────────────────────────────────────────────────────────────
auto_csv   = os.path.exists(DEFAULT_CSV)
auto_model = os.path.exists(DEFAULT_MODEL)

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding:18px 0 12px">
        <div style="font-size:2.2rem">🚦</div>
        <div style="font-size:1.05rem; font-weight:800; color:{C_RED}; margin:5px 0 2px">
            RiskRadar
        </div>
        <div style="font-size:0.72rem; color:{C_MUTED}; line-height:1.6">
            SLA Breach Risk<br>Decision Support
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section">📁 Data Source</div>', unsafe_allow_html=True)

    if auto_csv and auto_model:
        data_mode = st.radio(
            "data_mode",
            ["Pre-loaded dataset", "Upload your own"],
            label_visibility="collapsed",
        )
        if data_mode == "Pre-loaded dataset":
            st.success("✅ Demo dataset ready")
            uploaded   = None
            model_path = DEFAULT_MODEL
        else:
            uploaded   = st.file_uploader("Upload incident_event_log.csv", type=["csv"])
            model_path = st.text_input("Model path", value=DEFAULT_MODEL)
    else:
        data_mode  = "Upload your own"
        uploaded   = st.file_uploader("Upload incident_event_log.csv", type=["csv"])
        model_path = st.text_input("Model path", value=DEFAULT_MODEL)

    st.markdown("---")
    st.markdown('<div class="sidebar-section">🔍 Filters</div>', unsafe_allow_html=True)
    risk_filter = st.multiselect(
        "Risk band",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )
    min_prob = st.slider("Min probability", 0.0, 1.0, 0.0, 0.01)
    max_rows = st.slider("Max rows in table", 50, 5000, 500, 50)

    st.markdown("---")
    st.markdown('<div class="sidebar-section">↕️ Sort</div>', unsafe_allow_html=True)
    sort_by   = st.selectbox(
        "Sort by",
        ["sla_breach_probability", "reassignment_count_max", "reopen_count_max", "total_events"],
    )
    sort_desc = st.checkbox("Descending", value=True)


# ── Load data and model ────────────────────────────────────────────────────────
if data_mode == "Pre-loaded dataset":
    with st.spinner("Loading dataset and model..."):
        df_inc = build_from_path(DEFAULT_CSV)
        try:
            model = load_model(model_path)
        except Exception:
            model = train_fresh_model(DEFAULT_CSV)

elif uploaded is not None:
    with st.spinner("Processing your dataset..."):
        df_inc = build_from_upload(uploaded)
        try:
            model = load_model(model_path)
        except Exception:
            model = train_fresh_model(DEFAULT_CSV)

else:
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 70px 30px;
        border: 2px dashed rgba(255,255,255,0.12);
        border-radius: 14px;
        margin-top: 20px;
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(10px);
    ">
        <div style="font-size:3rem">📂</div>
        <h3 style="color:#e2e8f0; margin:14px 0 8px; font-weight:700">No data loaded yet</h3>
        <p style="color:{C_MUTED}; margin:0; font-size:0.95rem">
            Upload your <code>incident_event_log.csv</code> from the sidebar to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if "sla_breached" not in df_inc.columns:
    st.error("Could not find 'sla_breached'. The CSV needs a 'made_sla' column.")
    st.stop()

df_scored = score_incidents(model, df_inc)

try:
    clf_name = type(model.steps[-1][1]).__name__
except Exception:
    clf_name = type(model).__name__

# Sidebar model info
with st.sidebar:
    st.markdown("---")
    st.markdown('<div class="sidebar-section">ℹ️ Model Info</div>', unsafe_allow_html=True)
    n_features = st.session_state.get("n_features", "?")
    try:
        updated = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(model_path)))
    except Exception:
        updated = "unknown"
    st.markdown(
        f"**Type:** {clf_name}  \n"
        f"**Features:** {n_features:,}  \n"
        f"**Updated:** {updated}"
    )
    st.markdown("---")
    st.markdown(
        f'<div class="sidebar-note">'
        f"Built with scikit-learn + Streamlit<br>"
        f"Dataset: UCI ML Repository"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Aggregate stats ────────────────────────────────────────────────────────────
total_inc   = len(df_scored)
avg_prob    = float(df_scored["sla_breach_probability"].mean())
breach_rate = float(df_scored["sla_breached"].mean())
high_count  = int((df_scored["risk_band"] == "High").sum())
med_count   = int((df_scored["risk_band"] == "Medium").sum())
low_count   = int((df_scored["risk_band"] == "Low").sum())

band_counts = (
    df_scored["risk_band"].value_counts()
    .reindex(["High", "Medium", "Low"]).fillna(0)
)

source_label = "Pre-loaded demo dataset" if data_mode == "Pre-loaded dataset" else "Custom upload"

# Status strip
st.markdown(f"""
<div style="
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(222, 226, 230, 0.6);
    border-radius: 10px;
    padding: 10px 20px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
">
    <span style="color:{C_GREEN}; font-size:0.85rem; font-weight:700">●</span>
    <span style="font-size:0.83rem; color:#94a3b8">
        <b style="color:#e2e8f0">{total_inc:,}</b> incidents loaded &nbsp;·&nbsp;
        <b style="color:#e2e8f0">{clf_name}</b> &nbsp;·&nbsp;
        {source_label}
    </span>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📋 Incidents", "🧾 Incident Detail", "✅ Performance"])


# ─── Tab 1: Overview ──────────────────────────────────────────────────────────
with tab1:

    # KPI row — custom glass cards with per-card accent colour
    st.markdown(f"""
    <div style="display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin-bottom:26px">
      <div class="kpi-card" style="border-top-color:{C_BLUE}">
        <div class="kpi-label">📋 Total Incidents</div>
        <div class="kpi-value">{total_inc:,}</div>
      </div>
      <div class="kpi-card" style="border-top-color:{C_BLUE}">
        <div class="kpi-label">📈 Avg Risk Score</div>
        <div class="kpi-value">{avg_prob:.3f}</div>
      </div>
      <div class="kpi-card" style="border-top-color:{C_RED}">
        <div class="kpi-label">🔴 High Risk</div>
        <div class="kpi-value" style="color:{C_RED}">{high_count:,}</div>
        <div class="kpi-delta" style="color:{C_RED}">↑ {high_count/total_inc:.1%} of total</div>
      </div>
      <div class="kpi-card" style="border-top-color:{C_AMBER}">
        <div class="kpi-label">🟡 Medium Risk</div>
        <div class="kpi-value" style="color:{C_AMBER}">{med_count:,}</div>
        <div class="kpi-delta" style="color:{C_AMBER}">↑ {med_count/total_inc:.1%} of total</div>
      </div>
      <div class="kpi-card" style="border-top-color:{C_RED}">
        <div class="kpi-label">⚠️ Breach Rate</div>
        <div class="kpi-value">{breach_rate:.1%}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Risk distribution
    st.markdown(f"#### 📉 Risk Distribution")
    col_hist, col_bands = st.columns(2)

    with col_hist:
        st.markdown("**Predicted breach probability**")
        st.pyplot(plot_hist(
            df_scored["sla_breach_probability"],
            bins=20, xlabel="Breach probability",
        ))
        p10 = df_scored["sla_breach_probability"].quantile(0.10)
        p50 = df_scored["sla_breach_probability"].quantile(0.50)
        p90 = df_scored["sla_breach_probability"].quantile(0.90)
        st.caption(f"Median: {p50:.3f} · 10th to 90th percentile: {p10:.3f} to {p90:.3f}")

    with col_bands:
        st.markdown("**Incidents by risk band**")
        st.pyplot(plot_risk_bands(band_counts))
        st.caption(
            f"🔴 High: {int(band_counts.get('High', 0)):,} · "
            f"🟡 Medium: {int(band_counts.get('Medium', 0)):,} · "
            f"🟢 Low: {int(band_counts.get('Low', 0)):,}"
        )

    st.markdown("---")

    # Risk breakdown
    st.markdown(f"#### 🔍 Risk Breakdown")
    col_donut, col_heat = st.columns([1, 1.5])

    with col_donut:
        st.markdown("**Share by risk band**")
        st.pyplot(plot_donut(
            ["High", "Medium", "Low"],
            [int(band_counts.get("High", 0)),
             int(band_counts.get("Medium", 0)),
             int(band_counts.get("Low", 0))],
        ))

    with col_heat:
        st.markdown("**Priority vs risk band**")
        fig = plot_heatmap(df_scored)
        if fig is None:
            st.info("Priority data not available in this dataset.")
        else:
            st.pyplot(fig)

    st.markdown("---")

    # Top groups and categories — horizontal bar charts are easier to read with long labels
    st.markdown(f"#### 🏢 Where Risk Concentrates")
    col_grp, col_cat = st.columns(2)

    if "assignment_group" in df_scored.columns:
        with col_grp:
            st.markdown("**Top 10 riskiest assignment groups**")
            top_groups = (
                df_scored.groupby("assignment_group")["sla_breach_probability"]
                .mean().sort_values(ascending=True).tail(10)
            )
            st.pyplot(plot_hbar(
                top_groups.index.astype(str).tolist(),
                top_groups.values.tolist(),
                color=C_AMBER,
            ))

    if "category" in df_scored.columns:
        with col_cat:
            st.markdown("**Top 10 riskiest categories**")
            top_cats = (
                df_scored.groupby("category")["sla_breach_probability"]
                .mean().sort_values(ascending=True).tail(10)
            )
            st.pyplot(plot_hbar(
                top_cats.index.astype(str).tolist(),
                top_cats.values.tolist(),
                color=C_BLUE,
            ))

    st.markdown("---")

    # Calibration
    st.markdown(f"#### 📐 Calibration Check")
    st.caption("If the red line stays close to the dashed line, the predicted probabilities are reliable.")
    fig, calib = plot_calibration(df_scored)
    st.pyplot(fig)

    calib_show = calib.copy()
    calib_show["avg_pred"]    = calib_show["avg_pred"].round(3)
    calib_show["actual_rate"] = calib_show["actual_rate"].round(3)
    calib_show["n"]           = calib_show["n"].astype(int)
    with st.expander("Show calibration table"):
        st.dataframe(calib_show.reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.download_button(
        label="⬇️ Download all scored incidents as CSV",
        data=df_scored.to_csv(index=False).encode("utf-8"),
        file_name="riskradar_scored_incidents.csv",
        mime="text/csv",
    )


# ─── Tab 2: Incidents ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### 📋 Incident List")

    df_view = df_scored[df_scored["risk_band"].isin(risk_filter)].copy()
    df_view = df_view[df_view["sla_breach_probability"] >= min_prob]

    if sort_by in df_view.columns:
        df_view = df_view.sort_values(sort_by, ascending=not sort_desc)

    search_col, count_col = st.columns([3, 1])
    with search_col:
        q = st.text_input(
            "search", value="", label_visibility="collapsed",
            placeholder="🔎 Search by incident number...",
        )
    with count_col:
        st.markdown(
            f"<div style='padding-top:8px; color:{C_MUTED}; font-size:0.85rem'>"
            f"<b>{min(len(df_view), max_rows):,}</b> results</div>",
            unsafe_allow_html=True,
        )

    if q.strip():
        df_view = df_view[df_view["number"].astype(str).str.contains(q.strip(), case=False, na=False)]

    df_view = df_view.head(max_rows)

    show_cols = ["number", "sla_breach_probability", "risk_band", "recommended_action"]
    for col in ["priority", "assignment_group", "category", "subcategory",
                "total_events", "reassignment_count_max", "reopen_count_max", "resolution_hours"]:
        if col in df_view.columns:
            show_cols.append(col)

    st.dataframe(df_view[show_cols], use_container_width=True, height=480)
    st.caption("Tip: sort by breach probability and focus on High risk tickets first.")


# ─── Tab 3: Incident Detail ────────────────────────────────────────────────────
with tab3:
    st.markdown("#### 🧾 Incident Detail")

    ticket_ids = df_scored["number"].astype(str).tolist()
    chosen     = st.selectbox("Select an incident", ticket_ids, label_visibility="collapsed")

    row_df = df_scored[df_scored["number"].astype(str) == str(chosen)].iloc[:1].copy()
    row    = row_df.iloc[0]

    p      = float(row["sla_breach_probability"])
    band   = str(row["risk_band"])
    action = str(row["recommended_action"])

    band_color = {"High": C_RED, "Medium": C_AMBER, "Low": C_GREEN}.get(band, C_BLUE)
    band_icon  = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(band, "⚪")

    # Risk banner
    st.markdown(f"""
    <div style="
        border-left: 5px solid {band_color};
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid {band_color}40;
        border-left: 5px solid {band_color};
        border-radius: 12px;
        padding: 20px 26px;
        margin-bottom: 20px;
        box-shadow: 0 4px 22px rgba(0, 0, 0, 0.30);
    ">
        <p style="margin:0; color:{C_MUTED}; font-size:0.72rem;
                  text-transform:uppercase; letter-spacing:0.08em; font-weight:600">
            Incident {chosen}
        </p>
        <p style="margin:6px 0 4px; font-size:1.6rem; font-weight:800; color:{band_color}">
            {band_icon}&nbsp; {band} Risk
        </p>
        <p style="margin:0; font-size:1rem; color:#e2e8f0">
            Breach probability: <b style="color:{band_color}">{p:.1%}</b>
        </p>
        <p style="margin:10px 0 0; color:#94a3b8; font-size:0.88rem; font-style:italic">
            {action}
        </p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("**🔍 Risk Drivers**")
        for d in get_risk_drivers(row, df_scored)[:6]:
            st.markdown(f"&nbsp;&nbsp;• {d}")

    with right:
        st.markdown("**💬 Explanation**")

        def _val(col, cast=float):
            try:
                v = row[col]
                return cast(v) if pd.notna(v) else None
            except Exception:
                return None

        st.write(explain({
            "sla_breach_probability": p,
            "risk_level":             band,
            "recommended_action":     action,
            "key_factors": {
                "total_events":           _val("total_events", int),
                "reassignment_count_max": _val("reassignment_count_max", int),
                "reopen_count_max":       _val("reopen_count_max", int),
                "resolution_hours":       _val("resolution_hours", float),
                "priority":               _val("priority", str),
                "assignment_group":       _val("assignment_group", str),
                "category":               _val("category", str),
            },
        }))

    st.markdown("---")
    with st.expander("Full incident row"):
        st.dataframe(row_df, use_container_width=True)


# ─── Tab 4: Performance ───────────────────────────────────────────────────────
with tab4:
    st.markdown("#### ✅ Model Performance")
    st.caption("Adjust the threshold to see how precision and recall trade off on this dataset.")

    y_true  = df_scored["sla_breached"].astype(int).values
    y_prob  = df_scored["sla_breach_probability"].values
    auc_ok  = len(set(y_true.tolist())) == 2
    auc     = float(roc_auc_score(y_true, y_prob)) if auc_ok else None

    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
    y_pred    = (y_prob >= threshold).astype(int)

    prec    = float(precision_score(y_true, y_pred, zero_division=0))
    rec     = float(recall_score(y_true, y_pred, zero_division=0))
    f1      = float(f1_score(y_true, y_pred, zero_division=0))
    cm      = confusion_matrix(y_true, y_pred, labels=[0, 1])
    flagged = int(y_pred.sum())
    total   = int(len(y_pred))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Precision", f"{prec:.3f}")
    m2.metric("🔔 Recall",    f"{rec:.3f}")
    m3.metric("⚡ F1 Score",  f"{f1:.3f}")
    m4.metric("📈 ROC AUC",   f"{auc:.3f}" if auc is not None else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        f"At threshold **{threshold:.2f}**, the model flags **{flagged:,}** of "
        f"**{total:,}** incidents as breach risk."
    )

    cm_col, note_col = st.columns([1, 1.2])

    with cm_col:
        st.markdown("**Confusion Matrix**")
        st.pyplot(plot_confusion(cm, threshold))

    with note_col:
        st.markdown("**Reading the results**")
        st.markdown(f"""
        <div style="
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(91, 158, 201, 0.22);
            border-left: 4px solid {C_BLUE};
            border-radius: 12px;
            padding: 18px 22px;
            margin-top: 6px;
            font-size: 0.875rem;
            line-height: 1.80;
            color: #cbd5e1;
            box-shadow: 0 4px 22px rgba(0, 0, 0, 0.28);
        ">
            <b>🎯 Precision</b> — of all the tickets the model flagged,
            what share actually breached SLA.<br><br>
            <b>🔔 Recall</b> — of all real breaches in the data,
            what share the model managed to catch.<br><br>
            <b>⚡ F1</b> — harmonic mean of precision and recall.
            A balanced single score.<br><br>
            <b>💡 Tip</b> — lowering the threshold catches more real breaches
            at the cost of more false alarms. Raising it does the opposite.
            Adjust based on how much capacity the team has to investigate flagged tickets.
        </div>
        """, unsafe_allow_html=True)
