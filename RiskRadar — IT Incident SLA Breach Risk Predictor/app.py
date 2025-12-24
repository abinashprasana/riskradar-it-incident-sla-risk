import os
import time
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

load_dotenv()

from data_processing import load_event_log, build_incident_summary
from decision_logic import risk_band, recommended_action
from llm_explainer import explain

st.set_page_config(page_title="RiskRadar - IT Incident SLA Risk", layout="wide")

COL_MAIN = "#ff6b6b"
COL_ACCENT = "#4dabf7"
COL_OK = "#51cf66"
COL_WARN = "#fcc419"
COL_DARK = "#343a40"


# -----------------------------
# Loading + scoring
# -----------------------------
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


@st.cache_data
def build_summary_from_upload(file) -> pd.DataFrame:
    df_events = load_event_log(file)
    return build_incident_summary(df_events)


def score_all_incidents(model, df_inc: pd.DataFrame) -> pd.DataFrame:
    df = df_inc.copy()

    drop_cols = [c for c in ["number", "sla_breached", "made_sla", "opened_at_min", "resolved_at_max"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # for model info
    st.session_state["n_features_used"] = int(X.shape[1])

    probs = model.predict_proba(X)[:, 1]
    df["sla_breach_probability"] = probs
    df["risk_band"] = df["sla_breach_probability"].apply(risk_band)
    df["recommended_action"] = df["risk_band"].apply(recommended_action)

    return df.sort_values("sla_breach_probability", ascending=False).reset_index(drop=True)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# -----------------------------
# Plot helpers
# -----------------------------
def _new_fig():
    return plt.figure()


def plot_hist(series, bins=20, xlabel="", ylabel="", color=COL_ACCENT):
    fig = _new_fig()
    plt.hist(series, bins=bins, color=color, edgecolor="white", alpha=0.9)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.2)
    return fig


def plot_bar(x_labels, y_values, xlabel="", ylabel="", color=COL_MAIN, rotate=False):
    fig = _new_fig()
    plt.bar(x_labels, y_values, color=color, edgecolor="white", alpha=0.95)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if rotate:
        plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.2)
    return fig


def plot_donut(labels, values, title=""):
    fig = _new_fig()
    colors = [COL_MAIN, COL_WARN, COL_OK]
    plt.pie(
        values,
        labels=labels,
        colors=colors[: len(labels)],
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops={"width": 0.45, "edgecolor": "white"},
    )
    plt.title(title)
    return fig


def plot_heatmap_priority_risk(df_scored: pd.DataFrame):
    if "priority" not in df_scored.columns:
        return None

    pivot = (
        pd.crosstab(df_scored["priority"], df_scored["risk_band"])
        .reindex(columns=["High", "Medium", "Low"])
        .fillna(0)
    )

    fig = _new_fig()
    plt.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    plt.xlabel("Risk band")
    plt.ylabel("Priority")
    plt.title("Priority × Risk band (count)")
    plt.colorbar(label="Count")

    # auto text color for readability
    max_val = float(pivot.values.max()) if float(pivot.values.max()) > 0 else 1.0
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = float(pivot.values[i, j])
            ratio = val / max_val
            txt_color = "white" if ratio > 0.55 else "black"
            plt.text(
                j, i,
                f"{int(val)}",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=10,
                fontweight="bold"
            )

    return fig, pivot


def plot_calibration(df_scored: pd.DataFrame):
    tmp = df_scored.copy()
    tmp["prob_bin"] = pd.cut(tmp["sla_breach_probability"], bins=10)

    calib = tmp.groupby("prob_bin").agg(
        avg_pred=("sla_breach_probability", "mean"),
        actual_rate=("sla_breached", "mean"),
        n=("sla_breached", "count"),
    ).dropna()

    fig = _new_fig()
    plt.plot(calib["avg_pred"], calib["actual_rate"], marker="o", color=COL_MAIN, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color=COL_DARK, alpha=0.5)
    plt.xlabel("Avg predicted probability (bin)")
    plt.ylabel("Actual breach rate (bin)")
    plt.grid(alpha=0.2)

    return fig, calib


def plot_confusion(cm, labels=("0", "1"), title="Confusion Matrix"):
    fig = _new_fig()
    plt.imshow(cm, cmap="Blues", aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.colorbar(label="Count")

    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            txt_color = "white" if val > cm.max() * 0.6 else "black"
            plt.text(j, i, str(val), ha="center", va="center", color=txt_color, fontweight="bold")

    return fig


# -----------------------------
# Simple risk drivers (no heavy libs)
# -----------------------------
def get_risk_drivers(row: pd.Series, df_scored: pd.DataFrame):
    drivers = []

    def safe_float(x):
        try:
            if pd.isna(x):
                return None
            return float(x)
        except Exception:
            return None

    checks = [
        ("reassignment_count_max", "Lots of reassignments (handoffs can delay resolution)"),
        ("reopen_count_max", "Ticket reopened multiple times (unstable fix / recurring issue)"),
        ("resolution_hours", "Long resolution time pattern (slow closure behavior)"),
        ("total_events", "Many event updates (possible churn / frequent status changes)"),
    ]

    for col, msg in checks:
        if col not in df_scored.columns or col not in row.index:
            continue

        v = safe_float(row[col])
        if v is None:
            continue

        med = safe_float(df_scored[col].median())
        q75 = safe_float(df_scored[col].quantile(0.75))
        if med is None or q75 is None:
            continue

        if v >= q75 and v > med:
            drivers.append(f"{msg} — value={v:g} (above typical)")

    if "priority" in row.index and pd.notna(row["priority"]):
        pr = str(row["priority"])
        if "1" in pr or "Critical" in pr:
            drivers.append("Priority is Critical — usually needs faster escalation")
        elif "2" in pr or "High" in pr:
            drivers.append("Priority is High — worth keeping an eye")

    if not drivers:
        drivers.append("Nothing extreme pops out in the basic drivers — looks more like a normal-flow ticket.")

    return drivers


# -----------------------------
# UI
# -----------------------------
st.title("RiskRadar — IT Incident SLA Breach Risk (Decision Support)")
st.caption("ML predicts risk; explanations are generated only from computed facts.")

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload incident_event_log.csv", type=["csv"])
    model_path = st.text_input("Model path", value="best_model.joblib")

    st.divider()
    st.subheader("Filters")
    risk_filter = st.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    min_prob = st.slider("Min probability", 0.0, 1.0, 0.0, 0.01)
    max_rows = st.slider("Max rows in table", 50, 5000, 500, 50)

    st.divider()
    st.subheader("Sort")
    sort_by = st.selectbox("Sort by", ["sla_breach_probability", "reassignment_count_max", "reopen_count_max", "total_events"])
    sort_desc = st.checkbox("Descending", value=True)

if uploaded is None:
    st.info("Upload your CSV to start.")
    st.stop()

df_inc = build_summary_from_upload(uploaded)

if "sla_breached" not in df_inc.columns:
    st.error("Could not find 'sla_breached'. Your CSV needs a 'made_sla' column.")
    st.stop()

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

df_scored = score_all_incidents(model, df_inc)

# model info
with st.sidebar:
    st.divider()
    st.subheader("Model info")
    model_type = type(model).__name__
    n_features = st.session_state.get("n_features_used", None)
    try:
        ts = os.path.getmtime(model_path)
        model_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
    except Exception:
        model_time = "unknown"

    st.write(f"- Type: **{model_type}**")
    st.write(f"- Model file updated: **{model_time}**")
    st.write(f"- Features used: **{n_features if n_features is not None else 'unknown'}**")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Incident List", "🧾 Incident Detail", "✅ Model Evaluation"])


# -----------------------------
# Dashboard
# -----------------------------
with tab1:
    st.subheader("Overview")

    total_inc = len(df_scored)
    avg_prob = float(df_scored["sla_breach_probability"].mean())
    breach_rate = float(df_scored["sla_breached"].mean())

    high_count = int((df_scored["risk_band"] == "High").sum())
    med_count = int((df_scored["risk_band"] == "Medium").sum())
    low_count = int((df_scored["risk_band"] == "Low").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total incidents", f"{total_inc}")
    c2.metric("Avg breach probability", f"{avg_prob:.2f}")
    c3.metric("High risk", f"{high_count}")
    c4.metric("Medium risk", f"{med_count}")
    st.metric("Actual breach rate (from dataset)", f"{breach_rate:.2f}")

    st.info(
        f"**Quick read:** Out of **{total_inc}** incidents, **{high_count}** are **High risk** and **{med_count}** are **Medium**. "
        f"Average predicted risk is **{avg_prob:.2f}**, dataset breach rate is **{breach_rate:.2f}**."
    )

    st.divider()
    st.subheader("Visuals")
    st.caption(
        "The charts below give a quick picture of where SLA breach risk is concentrated, "
        "which teams or incident types are more exposed, and whether the model’s confidence makes sense."
    )

    # Row 1: Distribution + Risk counts
    left, right = st.columns(2)

    with left:
        st.markdown("**1) Risk probability distribution**")
        st.write(
            "This shows how predicted SLA breach risk is spread across all incidents. "
            "Values closer to 1 mean higher confidence that an incident may breach SLA."
        )

        fig = plot_hist(
            df_scored["sla_breach_probability"],
            bins=20,
            xlabel="Predicted SLA breach probability",
            ylabel="Number of incidents",
            color=COL_ACCENT,
        )
        st.pyplot(fig)

        p10 = float(df_scored["sla_breach_probability"].quantile(0.10))
        p50 = float(df_scored["sla_breach_probability"].quantile(0.50))
        p90 = float(df_scored["sla_breach_probability"].quantile(0.90))

        st.caption(
            f"Most incidents fall between {p10:.2f} and {p90:.2f}. "
            f"The median risk score is {p50:.2f}."
        )

    with right:
        st.markdown("**2) Risk band counts**")
        st.write(
            "This groups incidents into High, Medium, and Low risk bands. "
            "It helps teams quickly see how many tickets need attention."
        )

        band_counts = (
            df_scored["risk_band"]
            .value_counts()
            .reindex(["High", "Medium", "Low"])
            .fillna(0)
        )

        fig = plot_bar(
            band_counts.index.tolist(),
            band_counts.values.tolist(),
            xlabel="Risk band",
            ylabel="Number of incidents",
            color=COL_MAIN,
        )
        st.pyplot(fig)

        st.caption(
            f"High: {int(band_counts.get('High', 0))}, "
            f"Medium: {int(band_counts.get('Medium', 0))}, "
            f"Low: {int(band_counts.get('Low', 0))}"
        )

    # Row 2: Donut + Heatmap
    st.write("")
    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**3) Risk band share**")
        st.write(
            "Same information as the bar chart, shown as a percentage. "
            "This gives a quick sense of overall risk exposure."
        )

        labels = ["High", "Medium", "Low"]
        values = [
            int(band_counts.get("High", 0)),
            int(band_counts.get("Medium", 0)),
            int(band_counts.get("Low", 0)),
        ]

        fig = plot_donut(labels, values, title="Risk band share")
        st.pyplot(fig)

    with d2:
        st.markdown("**4) Priority vs risk band**")
        st.write(
            "This shows how SLA risk is distributed across priority levels. "
            "Darker cells indicate more incidents in that priority–risk combination."
        )

        heat = plot_heatmap_priority_risk(df_scored)
        if heat is None:
            st.info("Priority information is not available for this dataset.")
        else:
            fig, _ = heat
            st.pyplot(fig)

    # Row 3: Groups + Categories
    st.write("")
    colA, colB = st.columns(2)

    if "assignment_group" in df_scored.columns:
        with colA:
            st.markdown("**5) Assignment groups with higher risk**")
            st.write(
                "Assignment groups are ranked by their average predicted SLA breach risk. "
                "This helps identify teams that may need additional support or monitoring."
            )

            top_groups = (
                df_scored.groupby("assignment_group")["sla_breach_probability"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )

            fig = plot_bar(
                top_groups.index.astype(str).tolist(),
                top_groups.values.tolist(),
                xlabel="Assignment group",
                ylabel="Average risk score",
                color=COL_WARN,
                rotate=True,
            )
            st.pyplot(fig)

    if "category" in df_scored.columns:
        with colB:
            st.markdown("**6) Incident categories with higher risk**")
            st.write(
                "Shows which types of incidents historically tend to have higher SLA breach risk. "
                "Useful for preventive actions and capacity planning."
            )

            top_cats = (
                df_scored.groupby("category")["sla_breach_probability"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )

            fig = plot_bar(
                top_cats.index.astype(str).tolist(),
                top_cats.values.tolist(),
                xlabel="Category",
                ylabel="Average risk score",
                color=COL_OK,
                rotate=True,
            )
            st.pyplot(fig)

    # Calibration
    st.divider()
    st.subheader("Calibration check")
    st.write(
        "This compares predicted SLA breach probability with actual breach rates. "
        "If the curve stays close to the dashed line, the model’s confidence is reliable."
    )

    fig, calib = plot_calibration(df_scored)
    st.pyplot(fig)

    calib_show = calib.copy()
    calib_show["avg_pred"] = calib_show["avg_pred"].round(2)
    calib_show["actual_rate"] = calib_show["actual_rate"].round(2)
    calib_show["n"] = calib_show["n"].astype(int)

    st.dataframe(calib_show.reset_index(drop=True), use_container_width=True)

    st.caption(
        "Above the dashed line means the model is under-confident; below it means the model is over-confident."
    )

    st.divider()
    st.subheader("Download scored dataset")
    st.download_button(
        label="⬇️ Download predictions as CSV",
        data=to_csv_bytes(df_scored),
        file_name="riskradar_scored_incidents.csv",
        mime="text/csv",
    )


# -----------------------------
# Incident List
# -----------------------------
with tab2:
    st.subheader("Incident List (sortable / filterable)")

    df_view = df_scored.copy()
    df_view = df_view[df_view["risk_band"].isin(risk_filter)]
    df_view = df_view[df_view["sla_breach_probability"] >= min_prob]

    if sort_by in df_view.columns:
        df_view = df_view.sort_values(sort_by, ascending=not sort_desc)

    q = st.text_input("Search incident number (contains)", value="")
    if q.strip():
        df_view = df_view[df_view["number"].astype(str).str.contains(q.strip(), case=False, na=False)]

    df_view = df_view.head(max_rows)

    show_cols = ["number", "sla_breach_probability", "risk_band", "recommended_action"]
    for extra in [
        "priority",
        "assignment_group",
        "category",
        "subcategory",
        "total_events",
        "reassignment_count_max",
        "reopen_count_max",
        "resolution_hours",
    ]:
        if extra in df_view.columns:
            show_cols.append(extra)

    st.dataframe(df_view[show_cols], use_container_width=True)
    st.caption("Tip: sort by breach probability and focus on High risk first.")


# -----------------------------
# Incident Detail
# -----------------------------
with tab3:
    st.subheader("Incident Detail")

    ticket_ids = df_scored["number"].astype(str).tolist()
    chosen = st.selectbox("Pick an incident number", ticket_ids)

    row_df = df_scored[df_scored["number"].astype(str) == str(chosen)].iloc[0:1].copy()
    row = row_df.iloc[0]

    p = float(row["sla_breach_probability"])
    band = str(row["risk_band"])
    action = str(row["recommended_action"])

    left, right = st.columns(2)

    with left:
        st.subheader("Risk")
        st.metric("SLA breach probability", f"{p:.2f}")
        st.metric("Risk band", band)
        st.write("**Recommended action**")
        st.write(action)

        st.divider()
        st.markdown("**Top drivers (simple)**")
        drivers = get_risk_drivers(row, df_scored)
        for d in drivers[:6]:
            st.write(f"- {d}")

    with right:
        st.subheader("Explanation (LLM)")
        summary = {
            "sla_breach_probability": p,
            "risk_level": band,
            "recommended_action": action,
            "key_factors": {
                "total_events": int(row["total_events"]) if "total_events" in row_df.columns and pd.notna(row.get("total_events")) else None,
                "reassignment_count_max": int(row["reassignment_count_max"]) if "reassignment_count_max" in row_df.columns and pd.notna(row.get("reassignment_count_max")) else None,
                "reopen_count_max": int(row["reopen_count_max"]) if "reopen_count_max" in row_df.columns and pd.notna(row.get("reopen_count_max")) else None,
                "resolution_hours": float(row["resolution_hours"]) if "resolution_hours" in row_df.columns and pd.notna(row.get("resolution_hours")) else None,
                "priority": str(row["priority"]) if "priority" in row_df.columns and pd.notna(row.get("priority")) else None,
                "assignment_group": str(row["assignment_group"]) if "assignment_group" in row_df.columns and pd.notna(row.get("assignment_group")) else None,
                "category": str(row["category"]) if "category" in row_df.columns and pd.notna(row.get("category")) else None,
            },
        }
        st.write(explain(summary))

    st.divider()
    st.subheader("Incident summary row")
    st.dataframe(row_df, use_container_width=True)


# -----------------------------
# Model Evaluation
# -----------------------------
with tab4:
    st.subheader("Model Evaluation")
    st.caption("Basic performance + threshold behavior using the dataset labels (sla_breached).")

    y_true = df_scored["sla_breached"].astype(int).values
    y_prob = df_scored["sla_breach_probability"].values

    auc_ok = len(set(y_true.tolist())) == 2
    auc = float(roc_auc_score(y_true, y_prob)) if auc_ok else None

    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
    y_pred = (y_prob >= threshold).astype(int)

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    flagged = int(y_pred.sum())
    total = int(len(y_pred))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{prec:.2f}")
    c2.metric("Recall", f"{rec:.2f}")
    c3.metric("F1", f"{f1:.2f}")
    c4.metric("ROC AUC", f"{auc:.2f}" if auc is not None else "N/A")

    st.info(
        f"At threshold **{threshold:.2f}**, the model flags **{flagged}/{total}** incidents as 'breach risk'. "
        "Higher threshold = fewer alerts (more strict). Lower threshold = more alerts (more sensitive)."
    )

    st.write("**Confusion matrix (0 = no breach, 1 = breach)**")
    fig = plot_confusion(cm, labels=("0", "1"), title=f"Confusion Matrix @ threshold {threshold:.2f}")
    st.pyplot(fig)

    st.write("**Quick notes**")
    st.write("- Precision: how many flagged incidents were actually breaches.")
    st.write("- Recall: how many real breaches were successfully caught.")
    st.write("- Threshold is the control knob based on ops workload.")
