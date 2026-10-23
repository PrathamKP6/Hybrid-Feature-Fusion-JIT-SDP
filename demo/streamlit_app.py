"""Polished Streamlit interface for the offline Experiment 7F demo.

Run from the repository root:
    python -m streamlit run demo/streamlit_app.py

Displays saved predictions for validation/test splits; does not predict arbitrary
unseen commits.
"""
from pathlib import Path
import html
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.cli.app import build_results  # noqa: E402

st.set_page_config(
    page_title="Hybrid Feature Fusion | JIT Defect Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Distinct visual theme: navy sidebar, blue/cyan hero, elevated cards.
st.markdown(
    """
    <style>
    /* Consistent light theme: readable labels, controls, cards and charts. */
    :root { color-scheme: light; }
    .stApp {
      background: linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%);
      color: #172554;
    }
    [data-testid="stHeader"] { background: rgba(248,251,255,.98); }
    [data-testid="stAppViewContainer"] { background: transparent; }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #ffffff 0%, #f0f6ff 100%);
      border-right: 1px solid #d9e5f4;
    }
    [data-testid="stSidebar"] * { color: #172554 !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #172554 !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #475569 !important; }
    .hero {
      padding: 1.8rem 2rem; border-radius: 22px; color: #fff;
      background: linear-gradient(118deg, #10234b 0%, #1d4ed8 58%, #0891b2 100%);
      box-shadow: 0 12px 28px rgba(30,64,175,.14); margin-bottom: 1.4rem;
    }
    .hero-kicker { text-transform: uppercase; letter-spacing: .16em;
      font-size: .72rem; font-weight: 800; color: #dbeafe; }
    .hero h1 { color: #fff !important; margin: .35rem 0 .45rem; font-size: 2.25rem; line-height: 1.15; }
    .hero p { color: #eff6ff !important; margin: 0; font-size: 1rem; }
    .section-title { color: #14264d !important; font-size: 1.35rem; font-weight: 800; margin: .65rem 0 .8rem; }
    .muted, .subsection-note { color: #52627a !important; font-size: .9rem; }
    .feature-card {
      background: #fff; border: 1px solid #d8e4f2; border-top: 4px solid #3b82f6;
      border-radius: 17px; padding: 1.15rem 1.15rem 1.25rem; min-height: 170px;
      box-shadow: 0 6px 18px rgba(15,23,42,.045);
    }
    .feature-card.cyan { border-top-color: #0891b2; }
    .feature-card.violet { border-top-color: #7c3aed; }
    .feature-icon { font-size: 1.7rem; margin-bottom: .35rem; }
    .feature-title { font-weight: 800; color: #172554 !important; font-size: 1.03rem; }
    .feature-text { color: #475569 !important; font-size: .9rem; line-height: 1.52; margin-top: .45rem; }
    .info-strip { background: #eff6ff; border: 1px solid #bfdbfe; color: #1e3a8a !important;
      padding: .85rem 1rem; border-radius: 12px; font-size: .9rem; }
    .risk-high { background: #fff1f2; color: #9f1239 !important; border: 1px solid #fecdd3;
      padding: .75rem .9rem; border-radius: 12px; font-weight: 800; }
    .risk-low { background: #ecfdf5; color: #065f46 !important; border: 1px solid #a7f3d0;
      padding: .75rem .9rem; border-radius: 12px; font-weight: 800; }
    div[data-testid="stMetric"] { background: #fff; border: 1px solid #d8e4f2; border-radius: 15px;
      padding: .9rem 1rem; box-shadow: 0 5px 15px rgba(15,23,42,.04); }
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] label, div[data-testid="stMetricLabel"] div {
      color: #334155 !important; opacity: 1 !important; -webkit-text-fill-color: #334155 !important; font-weight: 700 !important;
    }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] div,
    div[data-testid="stMetricValue"] p { color: #10234b !important; opacity: 1 !important;
      -webkit-text-fill-color: #10234b !important; font-weight: 750 !important; }
    .metric-name { color: #334155 !important; font-size: .92rem; font-weight: 800; line-height: 1.35; margin-bottom: .45rem; }
    .metric-value { color: #10234b !important; font-size: 2rem; font-weight: 800; line-height: 1.15; letter-spacing: -.03em; }
    .metric-card { background: #fff; border: 1px solid #d8e4f2; border-radius: 15px; padding: 1rem 1.1rem;
      min-height: 112px; margin-bottom: .7rem; box-shadow: 0 5px 16px rgba(15,23,42,.04); }
    .metric-card, .metric-card * { text-shadow: none !important; filter: none !important; opacity: 1 !important; }
    .metric-card .metric-help { color: #52627a !important; font-size: .78rem; line-height: 1.35; margin-top: .35rem; }
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] div, [data-testid="stMarkdownContainer"] label {
      color: #334155 !important; opacity: 1 !important; -webkit-text-fill-color: #334155 !important;
    }
    [data-baseweb="select"] > div, [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div { background: #fff !important; border-color: #cbd5e1 !important; }
    [data-baseweb="select"] *, [data-baseweb="input"] input { color: #172554 !important; }
    .explorer-kicker { color: #64748b !important; font-size: .82rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
    div[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid #d8e4f2; border-radius: 12px; }
    .footer-note { color: #64748b; font-size: .8rem; margin-top: 1.3rem; border-top: 1px solid #dbe5f1; padding-top: .8rem; }
    /* Keep the page crisp at rest; Streamlit may show its native rerun indicator while loading. */
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🧬 HFF · JIT-SDP")
    st.caption("Research demonstration")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Commit explorer", "Model performance"],
        label_visibility="visible",
    )
    st.markdown("### Dataset split")
    split = st.selectbox("Choose split", ["test", "validation"], index=0)
    st.divider()
    st.markdown("**Experiment 7F**")
    st.caption("JIT metrics + CodeBERT + LLM reasoning features")
    st.caption("Classification threshold · 0.50")
    st.divider()
    st.caption("Offline demo · Saved models and frozen split predictions")


@st.cache_data(show_spinner="Loading saved predictions and commit metadata…")
def get_results(selected_split: str) -> pd.DataFrame:
    return build_results(selected_split)


try:
    results = get_results(split)
except Exception as exc:
    st.error("The demo could not load the saved predictions.")
    st.exception(exc)
    st.info(
        "Run this app from the repository root with the project's virtual "
        "environment active, and verify the saved model and data artifacts exist."
    )
    st.stop()

if results.empty:
    st.warning("No commit predictions were loaded for this split.")
    st.stop()

results = results.copy()
results["commit_id"] = results["commit_id"].astype(str)
results["blend_probability"] = pd.to_numeric(results["blend_probability"], errors="coerce")


def clear_commit_hash_search():
    """Clear the hash input through a callback before its widget is instantiated."""
    st.session_state["commit_hash_search"] = ""
results["blend_prediction"] = pd.to_numeric(
    results["blend_prediction"], errors="coerce"
).fillna(0).astype(int)


def normalize_label(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "1", "buggy", "bug"}:
            return 1
        if value in {"false", "0", "non-buggy", "non_buggy", "nonbuggy"}:
            return 0
        try:
            return int(float(value))
        except ValueError:
            return None
    try:
        return int(bool(value))
    except (TypeError, ValueError):
        return None


results["_actual_label"] = (
    results["buggy"].map(normalize_label) if "buggy" in results.columns else None
)

def label_name(value):
    return "Buggy" if int(value) == 1 else "Non-buggy"


def metric_card(label, value, help_text=None):
    """Render a high-contrast KPI card with an explicit, readable label."""
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    help_html = (f'<div class="metric-help">{html.escape(str(help_text))}</div>'
                 if help_text else "")
    st.markdown(
        f'<div class="metric-card"><div class="metric-name">{safe_label}</div>'
        f'<div class="metric-value">{safe_value}</div>{help_html}</div>',
        unsafe_allow_html=True,
    )


def show_commit_detail(row):
    probability = float(row["blend_probability"])
    prediction = int(row["blend_prediction"])
    st.markdown('<div class="section-title">Commit analysis</div>', unsafe_allow_html=True)
    detail_col, prediction_col = st.columns([1.3, 1])
    with detail_col:
        st.markdown("#### 🧾 Commit metadata")
        st.code(str(row.get("commit_id", "Unknown")), language=None)
        st.write("**Project:**", row.get("project", "Unknown"))
        st.write("**Year:**", row.get("year", "Unknown"))
        st.write("**Dataset split:**", split.title())
    with prediction_col:
        st.markdown("#### 🎯 Blended model estimate")
        metric_card("Estimated defect-risk score", f"{probability:.2%}",
                    "Blended estimate from Random Forest, XGBoost and LightGBM.")
        if prediction:
            st.markdown('<div class="risk-high">▲ Higher estimated defect risk</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low">● Lower estimated defect risk</div>', unsafe_allow_html=True)
        st.caption("Classification threshold: 50%")

    st.markdown("#### Individual model outputs")
    model_cols = st.columns(3)
    model_info = [
        ("🌲 Random Forest", "rf_probability"),
        ("⚡ XGBoost", "xgb_probability"),
        ("💡 LightGBM", "lightgbm_probability"),
    ]
    for col, (name, key) in zip(model_cols, model_info):
        if key in row.index and pd.notna(row[key]):
            with col:
                metric_card(name, f"{float(row[key]):.2%}")

    message = row.get("message", "")
    if pd.notna(message) and str(message).strip() and str(message).lower() != "nan":
        st.markdown("#### 💬 Commit message")
        st.info(str(message))
    actual = row.get("_actual_label")
    if pd.notna(actual):
        st.write("**Reference label:**", label_name(actual))
    st.caption(
        "These are estimates for a commit in the frozen dataset, not calibrated "
        "production probabilities or a guarantee that a defect exists."
    )


def recent_project_mix(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort newest years first and interleave projects efficiently.

    Uses vectorized group ranks rather than repeatedly slicing DataFrames one
    row at a time, which was a major source of Streamlit rerun latency.
    """
    if frame.empty:
        return frame.copy()

    ordered = frame.copy()
    if "year" not in ordered.columns:
        return ordered

    ordered["_year_sort"] = pd.to_numeric(ordered["year"], errors="coerce")
    if "project" in ordered.columns:
        ordered["_project_sort"] = ordered["project"].fillna("Unknown project").astype(str)
    else:
        ordered["_project_sort"] = "Unknown project"

    # Within each year, rank each project's commits. Sorting by this rank puts
    # the first commit from every project before the second commit from any
    # project, while preserving newest-year-first ordering.
    tie_breakers = ["commit_id"] if "commit_id" in ordered.columns else []
    ordered = ordered.sort_values(
        ["_year_sort", "_project_sort"] + tie_breakers,
        ascending=[False, True] + [True] * len(tie_breakers),
        na_position="last",
        kind="stable",
    )
    ordered["_project_year_rank"] = ordered.groupby(
        ["_year_sort", "_project_sort"], dropna=False, sort=False
    ).cumcount()
    ordered = ordered.sort_values(
        ["_year_sort", "_project_year_rank", "_project_sort"],
        ascending=[False, True, True],
        na_position="last",
        kind="stable",
    )
    return ordered.drop(
        columns=["_year_sort", "_project_sort", "_project_year_rank"],
        errors="ignore",
    ).reset_index(drop=True)


def correct_example_sample(frame: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Choose illustrative correctly classified examples, newest years first.

    This only curates the Overview examples. The complete evaluation metrics and
    confusion matrix remain unchanged on the Model performance page.
    """
    labeled = frame[frame["_actual_label"].notna()].copy()
    if labeled.empty:
        return labeled
    labeled["_actual_label"] = labeled["_actual_label"].astype(int)
    labeled = labeled[labeled["blend_prediction"].astype(int) == labeled["_actual_label"]]
    labeled = recent_project_mix(labeled)
    if labeled.empty:
        return labeled
    # Prefer broad project coverage before taking additional examples.
    if "project" not in labeled.columns:
        return labeled.head(limit)
    first_per_project = labeled.drop_duplicates("project", keep="first")
    selected = first_per_project.head(limit)
    if len(selected) < limit:
        remaining = labeled[~labeled.index.isin(selected.index)]
        selected = pd.concat([selected, remaining.head(limit - len(selected))])
    return recent_project_mix(selected).head(limit)


# Keep recent commits prominent and interleave projects in browseable tables.
results = recent_project_mix(results)

# Page header
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-kicker">Software engineering · Machine learning</div>
      <h1>Hybrid Feature Fusion Framework</h1>
      <p>Just-in-Time Software Defect Prediction &nbsp;·&nbsp; Experiment 7F &nbsp;·&nbsp; {split.title()} split</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Overview":
    total = len(results)
    high_risk = int((results["blend_prediction"] == 1).sum())
    low_risk = total - high_risk
    high_share = high_risk / total if total else 0
    labeled = results[results["_actual_label"].notna()]
    actual_buggy = int((labeled["_actual_label"] == 1).sum())
    actual_non_buggy = int((labeled["_actual_label"] == 0).sum())
    unlabeled = total - len(labeled)

    st.markdown('<div class="section-title">Dataset snapshot</div>', unsafe_allow_html=True)
    st.markdown('<div class="subsection-note">Counts for the selected split. Actual labels come from the dataset; risk classes are predictions made by the saved model.</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Total commits evaluated", f"{total:,}", f"{split.title()} split")
    with k2:
        metric_card("Actual buggy commits", f"{actual_buggy:,}", "Ground-truth label: Buggy")
    with k3:
        metric_card("Actual non-buggy commits", f"{actual_non_buggy:,}", "Ground-truth label: Non-buggy")
    with k4:
        metric_card("Commits with reference labels", f"{len(labeled):,}", f"{unlabeled:,} without a usable label" if unlabeled else "All commits have reference labels")

    st.markdown('<div class="section-title">Model prediction summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="subsection-note">These are predicted risk categories, not the actual buggy/non-buggy counts above.</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        metric_card("Predicted higher risk", f"{high_risk:,}", "Blended score ≥ 50%")
    with p2:
        metric_card("Predicted lower risk", f"{low_risk:,}", "Blended score < 50%")
    with p3:
        metric_card("Predicted higher-risk share", f"{high_share:.1%}", "Share of all commits in this split")

    st.markdown('<div class="section-title">Three complementary feature groups</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="feature-card"><div class="feature-icon">📊</div>'
            '<div class="feature-title">Traditional JIT metrics</div>'
            '<div class="feature-text">12 change metrics describing change size, complexity, and developer/change history.</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="feature-card cyan"><div class="feature-icon">🧠</div>'
            '<div class="feature-title">CodeBERT semantic features</div>'
            '<div class="feature-text">384 PCA-reduced embedding features that represent semantic information in code changes.</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="feature-card violet"><div class="feature-icon">✨</div>'
            '<div class="feature-title">LLM reasoning features</div>'
            '<div class="feature-text">51 encoded categorical and numerical confidence/margin features from the LLM reasoning stage.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Predicted risk-class distribution")
        st.caption("These bars show model predictions, not ground-truth labels.")
        pred_counts = pd.DataFrame({
            "Prediction class": ["Higher risk", "Lower risk"],
            "Number of commits": [high_risk, low_risk],
        })
        prediction_chart = alt.Chart(pred_counts).mark_bar(color="#3b82f6", cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("Prediction class:N", sort=["Higher risk", "Lower risk"], title=None,
                    axis=alt.Axis(labelAngle=0, labelColor="#334155", titleColor="#334155")),
            y=alt.Y("Number of commits:Q", title="Number of commits",
                    axis=alt.Axis(labelColor="#334155", titleColor="#334155", gridColor="#e2e8f0")),
            tooltip=["Prediction class", "Number of commits"],
        ).properties(height=260, background="#ffffff").configure_view(stroke="#d8e4f2")
        st.altair_chart(prediction_chart, use_container_width=True)
    with right:
        st.markdown("#### Demo walkthrough")
        st.markdown(
            "1. **Commit explorer** — search by hash or filter by project.\n"
            "2. **Commit analysis** — compare the blended score with individual model outputs.\n"
            "3. **Model performance** — review held-out evaluation metrics."
        )
        st.markdown(
            '<div class="info-strip">This interface displays saved predictions for the selected dataset split. It does not process arbitrary new commits.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### Illustrative sample predictions")
    st.caption(
        "These examples are correctly classified commits selected to demonstrate the interface. "
        "This is not a summary of all predictions; the full results, including false positives "
        "and false negatives, remain reflected in the Model performance page."
    )
    sample_cols = [
        col for col in [
            "commit_id", "project", "year", "blend_probability",
            "blend_prediction", "_actual_label"
        ] if col in results.columns
    ]
    sample = correct_example_sample(results, limit=10)[sample_cols].copy()
    if sample.empty:
        st.info("No correctly classified labelled examples are available in this split.")
    else:
        sample["blend_probability"] = sample["blend_probability"].map(
            lambda value: f"{value:.2%}" if pd.notna(value) else "N/A"
        )
        sample["blend_prediction"] = sample["blend_prediction"].map(
            {1: "Higher risk", 0: "Lower risk"}
        )
        sample["_actual_label"] = sample["_actual_label"].map(
            {1: "Buggy (1)", 0: "Non-buggy (0)"}
        )
        sample = sample.rename(columns={
            "commit_id": "Commit hash",
            "project": "Project",
            "year": "Year",
            "blend_probability": "Estimated risk",
            "blend_prediction": "Predicted class",
            "_actual_label": "Actual reference label",
        })
        st.dataframe(sample, use_container_width=True, hide_index=True)

elif page == "Commit explorer":
    st.markdown('<div class="section-title">Commit explorer</div>', unsafe_allow_html=True)
    st.caption(
        "Explore the saved Experiment 7F predictions. Results are ordered with the newest years first "
        "and projects interleaved to make the dataset easier to browse."
    )

    search_col, risk_col = st.columns([1.45, 1])
    with search_col:
        search = st.text_input(
            "Search commit hash", placeholder="Enter a full or partial commit hash…",
            help="Partial hashes are supported.", key="commit_hash_search"
        )
    with risk_col:
        risk_filter = st.selectbox(
            "Predicted risk group",
            ["All predictions", "Higher estimated risk", "Lower estimated risk"],
        )
    projects = (
        sorted(results["project"].dropna().astype(str).unique())
        if "project" in results else []
    )
    selected_projects = st.multiselect(
        "Choose project", options=projects, default=[], placeholder="Choose project",
        help="Leave this empty to include commits from all projects."
    )

    filtered = results.copy()
    if search.strip():
        filtered = filtered[
            filtered["commit_id"].str.contains(search.strip(), case=False, regex=False, na=False)
        ]
    if selected_projects and "project" in filtered.columns:
        filtered = filtered[filtered["project"].astype(str).isin(selected_projects)]
    if risk_filter == "Higher estimated risk":
        filtered = filtered[filtered["blend_prediction"] == 1]
    elif risk_filter == "Lower estimated risk":
        filtered = filtered[filtered["blend_prediction"] == 0]
    filtered = recent_project_mix(filtered)

    count_col, reset_col = st.columns([3, 1])
    with count_col:
        st.markdown(f"### {len(filtered):,} matching commits")
        st.caption("The table shows model estimates. Reference labels are available when inspecting a commit.")
    with reset_col:
        st.button(
            "Clear hash search",
            use_container_width=True,
            on_click=clear_commit_hash_search,
        )

    if filtered.empty:
        st.warning("No commits match those filters. Try a shorter hash or choose a different project.")
    else:
        # Sending thousands of rows to the browser on every widget rerun makes
        # the explorer sluggish. Keep filtering/counts over the full split, but
        # render a bounded recent page; users can narrow with hash/project/risk.
        table_limit = 300
        table_records = filtered.head(table_limit).reset_index(drop=True).copy()
        display = table_records.copy()
        display["blend_probability"] = display["blend_probability"].map(
            lambda value: f"{value:.2%}" if pd.notna(value) else "N/A"
        )
        display["blend_prediction"] = display["blend_prediction"].map(
            {1: "Higher risk", 0: "Lower risk"}
        )
        display_cols = [
            col for col in ["commit_id", "project", "year", "blend_probability", "blend_prediction"]
            if col in display.columns
        ]
        table_view = display[display_cols].rename(columns={
            "commit_id": "Commit hash",
            "project": "Project",
            "year": "Year",
            "blend_probability": "Estimated risk",
            "blend_prediction": "Predicted class",
        })
        # Make a selected table row drive the commit analysis below.
        table_key = "explorer_table_" + str(abs(hash((search.strip(), risk_filter, tuple(selected_projects), split))))
        st.markdown("#### Select a row to inspect that commit")
        if len(filtered) > table_limit:
            st.caption(f"Showing the {table_limit} most recent matching commits out of {len(filtered):,}. Use the filters above to narrow the list.")
        else:
            st.caption("Click any row in the table. The commit analysis below updates to the selected row.")
        event = st.dataframe(
            table_view,
            use_container_width=True,
            hide_index=True,
            height=390,
            on_select="rerun",
            selection_mode="single-row",
            key=table_key,
        )
        selected_rows = []
        try:
            selected_rows = list(event.selection.rows)
        except (AttributeError, TypeError):
            pass
        selected_position = selected_rows[0] if selected_rows else 0
        selected_position = min(max(int(selected_position), 0), len(table_records) - 1)
        selected_row = table_records.iloc[selected_position]

        st.divider()
        st.markdown("#### 🔎 Selected commit analysis")
        with st.spinner("Loading analysis for the selected commit…"):
            show_commit_detail(selected_row)

elif page == "Model performance":
    st.markdown('<div class="section-title">Experiment 7F · Evaluation dashboard</div>', unsafe_allow_html=True)
    st.caption("Metrics are calculated from the selected split's reference labels and saved blended predictions.")
    labeled = results[results["_actual_label"].notna()].copy()
    if labeled.empty:
        st.warning("Reference labels are not available in this split, so evaluation metrics cannot be calculated.")
    else:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            matthews_corrcoef, roc_auc_score, average_precision_score,
            confusion_matrix,
        )
        y_true = labeled["_actual_label"].astype(int).to_numpy()
        y_pred = labeled["blend_prediction"].astype(int).to_numpy()
        y_prob = labeled["blend_probability"].astype(float).to_numpy()

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-score": f1_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }
        try:
            metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
            metrics["PR-AUC (Average Precision)"] = average_precision_score(y_true, y_prob)
        except ValueError:
            pass

        st.markdown("#### Evaluation metrics")
        st.markdown('<div class="subsection-note">Scores use the selected split. Precision, recall and F1 use the 0.50 classification threshold; ROC-AUC and PR-AUC use continuous risk scores.</div>', unsafe_allow_html=True)
        metric_help = {
            "Accuracy": "Fraction of all commits classified correctly.",
            "Precision": "Of commits predicted buggy, the fraction actually buggy.",
            "Recall": "Of actual buggy commits, the fraction detected.",
            "F1-score": "Harmonic mean of precision and recall.",
            "MCC": "Balanced correlation measure from −1 to +1.",
            "ROC-AUC": "Ability to rank buggy commits above non-buggy commits.",
            "PR-AUC (Average Precision)": "Precision–recall ranking quality; useful with class imbalance.",
        }
        items = list(metrics.items())
        for start in range(0, len(items), 4):
            metric_cols = st.columns(4)
            for col, (name, value) in zip(metric_cols, items[start:start + 4]):
                with col:
                    metric_card(name, f"{value:.4f}", metric_help.get(name))

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_frame = pd.DataFrame(
            cm,
            index=["Actual non-buggy", "Actual buggy"],
            columns=["Predicted lower risk", "Predicted higher risk"],
        )
        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("#### Confusion matrix")
            styled_cm = cm_frame.style.format("{:,.0f}").set_properties(
                **{"text-align": "center", "font-size": "15px", "padding": "10px"}
            ).set_table_styles([
                {"selector": "th", "props": [("background-color", "#e8effb"),
                                                   ("color", "#172554"),
                                                   ("font-weight", "700"),
                                                   ("text-align", "center")]},
                {"selector": "td", "props": [("background-color", "#ffffff"),
                                                   ("color", "#172554"),
                                                   ("border", "1px solid #dbe5f1")]},
                {"selector": "table", "props": [("border-collapse", "collapse"),
                                                     ("width", "100%"),
                                                     ("border-radius", "10px")]},
            ])
            st.table(styled_cm)
            st.caption("Rows are actual labels; columns are predicted risk classes.")
        with right:
            st.markdown("#### Estimated risk-score distribution")
            st.caption("Counts in each estimated probability band, not actual defect counts.")
            # Use NumPy histogram counts instead of Categorical.value_counts(sort=...),
            # which is unsupported by some pandas versions bundled with student setups.
            bin_edges = np.linspace(0.0, 1.0, 11)
            bin_labels = [f"{i / 10:.1f}–{(i + 1) / 10:.1f}" for i in range(10)]
            counts, _ = np.histogram(y_prob, bins=bin_edges)
            chart_data = pd.DataFrame({
                "Risk range": bin_labels,
                "Number of commits": counts.astype(int),
            })
            risk_chart = alt.Chart(chart_data).mark_bar(color="#38a3db", cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("Risk range:N", sort=bin_labels, title="Estimated risk-score range",
                        axis=alt.Axis(labelAngle=0, labelColor="#334155", titleColor="#334155", labelOverlap=False)),
                y=alt.Y("Number of commits:Q", title="Number of commits",
                        axis=alt.Axis(labelColor="#334155", titleColor="#334155", gridColor="#e2e8f0")),
                tooltip=["Risk range", "Number of commits"],
            ).properties(height=300, background="#ffffff").configure_view(stroke="#d8e4f2")
            st.altair_chart(risk_chart, use_container_width=True)
            st.caption("Each bar counts commits whose estimated risk falls in that score range.")

        st.markdown("#### Actual label distribution")
        label_counts = pd.DataFrame({
            "Reference label": ["Non-buggy (0)", "Buggy (1)"],
            "Number of commits": [int((y_true == 0).sum()), int((y_true == 1).sum())],
        })
        label_cols = st.columns([1, 2])
        with label_cols[0]:
            st.dataframe(label_counts, use_container_width=True, hide_index=True)
        with label_cols[1]:
            st.markdown('<div class="info-strip">Actual buggy/non-buggy counts come from reference labels. Higher-risk/lower-risk counts come from model predictions. A high-risk prediction can be a false alarm, and a low-risk prediction can miss a buggy commit.</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="info-strip">Evaluation sample: {len(labeled):,} labeled commits · '
            f'{split.title()} split · classification threshold 0.50. Use the test split for final held-out reporting.</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer-note">Hybrid Feature Fusion Framework · Offline research demo · '
    'Risk estimates are not calibrated production probabilities.</div>',
    unsafe_allow_html=True,
)
