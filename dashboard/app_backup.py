"""
dashboard/app.py
================
FraudGuard ML — Analytics Dashboard
Five pages: Overview, Live Prediction, Model Analysis, Alert Feed, Batch Scoring.
Run: streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

st.set_page_config(
    page_title="FraudGuard ML",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal, safe CSS — only touches things Streamlit cannot do natively.
# No custom number rendering. No font-variant tricks. No complex HTML tables.
st.markdown("""
<style>
  .block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px;
  }

  section[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #21262d !important;
  }

  section[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
  }

  section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid #30363d !important;
    color: #8b949e !important;
    font-size: 0.8rem !important;
    border-radius: 6px !important;
  }

  section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #8b949e !important;
    color: #c9d1d9 !important;
  }

  .tier-badge {
    display: inline-block;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 12px;
  }
  .badge-critical { background: #ffebe9; color: #cf222e; }
  .badge-high     { background: #fff1e5; color: #bc4c00; }
  .badge-medium   { background: #fff8c5; color: #9a6700; }
  .badge-low      { background: #dafbe1; color: #1a7f37; }
  .badge-unknown  { background: #f6f8fa; color: #57606a; }

  .alert-item {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 11px 14px;
    margin-bottom: 5px;
  }
  .alert-item.critical { border-left: 3px solid #cf222e; }
  .alert-item.high     { border-left: 3px solid #bc4c00; }
  .alert-item.medium   { border-left: 3px solid #9a6700; }
  .alert-item.low      { border-left: 3px solid #1a7f37; }

  .mono {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 0.8rem;
    color: #0d1117;
  }

  .small-muted {
    font-size: 0.72rem;
    color: #57606a;
    margin-top: 2px;
  }

  .infobox {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.82rem;
    color: #24292f;
    line-height: 1.65;
  }

  .infobox code {
    font-family: "SFMono-Regular", Consolas, monospace;
    background: #e8ecf0;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 0.78rem;
  }

  .verdict-fraud {
    background: #fff0ee;
    border: 1px solid #f5b8b8;
    border-radius: 10px;
    padding: 24px;
    text-align: center;
  }

  .verdict-legit {
    background: #f0fdf4;
    border: 1px solid #b3e6c8;
    border-radius: 10px;
    padding: 24px;
    text-align: center;
  }

  .verdict-eyebrow {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #57606a;
    margin-bottom: 6px;
  }

  .verdict-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #0d1117;
    letter-spacing: -0.01em;
  }

  .verdict-score-fraud {
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 3rem;
    font-weight: 600;
    color: #cf222e;
    line-height: 1.1;
    margin: 6px 0;
  }

  .verdict-score-legit {
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 3rem;
    font-weight: 600;
    color: #1a7f37;
    line-height: 1.1;
    margin: 6px 0;
  }

  .gauge-track {
    background: #eaeef2;
    border-radius: 6px;
    height: 10px;
    margin: 10px 0 4px;
    overflow: hidden;
  }

  div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 14px 16px !important;
  }

  div[data-testid="stMetric"] label {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: #57606a !important;
  }

  div[data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: #0d1117 !important;
  }

  hr { border-color: #d0d7de !important; border-width: 0.5px !important; }

  .section-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #57606a;
    margin-bottom: 12px;
    margin-top: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #eaeef2;
  }
</style>
""", unsafe_allow_html=True)

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL    = "http://localhost:8000"
MODEL_DIR  = Path("models")
LOGS_DIR   = Path("logs")
ALERT_LOG  = LOGS_DIR / "fraud_alerts.jsonl"
REPORT_DIR = Path("reports/figures")

TIER_COLOR = {
    "CRITICAL": "#cf222e",
    "HIGH":     "#bc4c00",
    "MEDIUM":   "#9a6700",
    "LOW":      "#1a7f37",
}

FEATURE_NOTES: dict[str, str] = {
    "V14": "Strongest fraud signal — large negative values are highly predictive",
    "V12": "Second strongest — negative values correlate with fraud",
    "V4":  "Positive values associated with fraud",
    "V10": "Negative values flag suspicious activity",
    "V11": "Positive values correlate with fraud risk",
    "V17": "Interaction with V1 amplifies fraud signal",
}

_EXAMPLE_LEGIT = {
    "V1": -1.3598, "V2": -0.0728, "V3":  2.5364, "V4":  1.3782,
    "V5": -0.3383, "V6":  0.4624, "V7":  0.2396, "V8":  0.0987,
    "V9":  0.3638, "V10":-0.0902, "V11":-0.5516, "V12":-0.6178,
    "V13":-0.9914, "V14":-0.3114, "V15":  1.4682, "V16":-0.4704,
    "V17": 0.2079, "V18": 0.0258, "V19":  0.4039, "V20":  0.2514,
    "V21":-0.0183, "V22": 0.2778, "V23":-0.1105,  "V24":  0.0669,
    "V25": 0.1285, "V26":-0.1891, "V27":  0.1336, "V28": -0.0211,
    "Amount": 149.62, "Time": 406.0,
}
_EXAMPLE_FRAUD = {
    "V1": -2.3122, "V2":  1.9522, "V3": -1.6097, "V4":  3.9979,
    "V5": -0.5222, "V6": -1.4265, "V7": -2.5374, "V8":  1.3912,
    "V9": -2.7700, "V10":-2.7722, "V11":  3.2020, "V12":-2.8998,
    "V13": 1.0750, "V14":-0.6677, "V15":  0.1799, "V16":-0.4523,
    "V17":-0.5828, "V18":-0.0728, "V19":  0.6669, "V20":  0.1285,
    "V21": 0.4260, "V22":  0.5420, "V23":  0.2400, "V24":  0.0500,
    "V25": 0.1200, "V26":-0.2000, "V27":  0.1800, "V28": -0.0300,
    "Amount": 239.93, "Time": 150.0,
}


# ── Data helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def _load_model_metadata(model_name: str = "xgboost_model") -> dict:
    path = MODEL_DIR / f"{model_name}_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=10)
def _load_alert_log(max_rows: int = 500) -> pd.DataFrame:
    if not ALERT_LOG.exists():
        return pd.DataFrame()
    rows: list[dict] = []
    with open(ALERT_LOG) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows[-max_rows:])
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df


@st.cache_data(ttl=60)
def _load_model_results() -> pd.DataFrame:
    path = Path("reports/model_results.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _api_health() -> dict:
    if not REQUESTS_OK:
        return {"status": "requests_missing", "model_loaded": False}
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"status": "unreachable", "model_loaded": False}


def _api_predict(features: dict, explain: bool = False) -> Optional[dict]:
    if not REQUESTS_OK:
        return None
    try:
        params = {"explain": "true"} if explain else {}
        resp = requests.post(
            f"{API_URL}/predict", json=features, params=params, timeout=8
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _mpl_style() -> None:
    if not MPL_OK:
        return
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          9,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.7,
        "axes.edgecolor":     "#d0d7de",
        "axes.facecolor":     "#ffffff",
        "figure.facecolor":   "#ffffff",
        "xtick.color":        "#57606a",
        "ytick.color":        "#57606a",
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "text.color":         "#24292f",
        "grid.color":         "#eaeef2",
        "grid.linewidth":     0.5,
    })


def _tier_badge(tier: str) -> str:
    css = {
        "CRITICAL": "badge-critical",
        "HIGH":     "badge-high",
        "MEDIUM":   "badge-medium",
        "LOW":      "badge-low",
    }.get(tier, "badge-unknown")
    return f'<span class="tier-badge {css}">{tier}</span>'


def _divider() -> None:
    st.markdown("<hr>", unsafe_allow_html=True)


def _section(label: str) -> None:
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<div style="padding: 24px 16px 16px;">'
            '<div style="font-size:1.05rem;font-weight:700;color:#f0f6fc;'
            'letter-spacing:-0.015em;">FraudGuard ML</div>'
            '<div style="font-size:0.68rem;color:#484f58;margin-top:2px;'
            'text-transform:uppercase;letter-spacing:0.07em;">Platform v2.1</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="height:1px;background:#21262d;margin:0 16px 16px;"></div>',
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            ["Overview", "Live Prediction", "Model Analysis", "Alert Feed", "Batch Scoring"],
            label_visibility="collapsed",
        )

        st.markdown(
            '<div style="height:1px;background:#21262d;margin:16px 16px;"></div>',
            unsafe_allow_html=True,
        )

        health   = _api_health()
        api_ok   = health.get("status") == "ok"
        model_ok = health.get("model_loaded", False)

        st.markdown(
            '<div style="padding:0 16px;font-size:0.65rem;font-weight:600;'
            'text-transform:uppercase;letter-spacing:0.09em;color:#484f58;'
            'margin-bottom:10px;">System Status</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        api_color = "#3fb950" if api_ok   else "#f85149"
        mod_color = "#58a6ff" if model_ok else "#484f58"
        api_txt   = "Online"  if api_ok   else "Offline"
        mod_txt   = "Ready"   if model_ok else "Missing"

        c1.markdown(
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:6px;'
            f'padding:9px 11px;">'
            f'<div style="font-size:0.6rem;color:#484f58;text-transform:uppercase;'
            f'letter-spacing:0.07em;margin-bottom:3px;">API</div>'
            f'<div style="font-size:0.85rem;font-weight:600;color:{api_color};">{api_txt}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        c2.markdown(
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:6px;'
            f'padding:9px 11px;">'
            f'<div style="font-size:0.6rem;color:#484f58;text-transform:uppercase;'
            f'letter-spacing:0.07em;margin-bottom:3px;">Model</div>'
            f'<div style="font-size:0.85rem;font-weight:600;color:{mod_color};">{mod_txt}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        uptime = health.get("uptime_seconds", 0)
        if uptime:
            h, r = divmod(int(uptime), 3600)
            m, s = divmod(r, 60)
            st.caption(f"Uptime  {h:02d}:{m:02d}:{s:02d}")

        mon = health.get("monitor", {})
        if mon.get("fraud_rate_5min") is not None:
            st.caption(
                f"Fraud rate  {mon['fraud_rate_5min']:.2%}   "
                f"P99  {mon.get('latency_p99_ms', 0):.0f} ms"
            )

        st.markdown(
            '<div style="height:1px;background:#21262d;margin:14px 16px;"></div>',
            unsafe_allow_html=True,
        )

        meta = _load_model_metadata()
        st.markdown(
            '<div style="padding:0 16px;font-size:0.65rem;font-weight:600;'
            'text-transform:uppercase;letter-spacing:0.09em;color:#484f58;'
            'margin-bottom:8px;">Active Model</div>',
            unsafe_allow_html=True,
        )
        if meta:
            model_class = meta.get("class", "Unknown")
            pr_val      = meta.get("val_pr_auc")
            st.markdown(
                f'<div style="padding:0 16px;">'
                f'<div style="font-family:monospace;font-size:0.75rem;color:#58a6ff;">'
                f'{model_class}</div>'
                + (
                    f'<div style="font-size:1.3rem;font-weight:700;color:#f0f6fc;'
                    f'margin-top:4px;font-family:monospace;">{float(pr_val):.4f}'
                    f'<span style="font-size:0.62rem;color:#484f58;'
                    f'font-weight:400;margin-left:6px;font-family:sans-serif;">PR-AUC</span>'
                    f'</div>'
                    if pr_val else ""
                )
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="padding:0 16px;font-size:0.78rem;color:#484f58;">'
                'No model found.<br>Run: <span style="font-family:monospace;'
                'color:#58a6ff;">python main.py</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div style="height:1px;background:#21262d;margin:14px 16px 12px;"></div>',
            unsafe_allow_html=True,
        )

        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    return page


# ── Page 1 — Overview ──────────────────────────────────────────────────────────

def _page_overview() -> None:
    st.title("Transaction Overview")
    st.caption(f"Last updated  {time.strftime('%H:%M:%S UTC', time.gmtime())}")

    alert_df = _load_alert_log(1_000)

    total_alerts   = len(alert_df)
    critical_count = int((alert_df["risk_tier"] == "CRITICAL").sum()) if not alert_df.empty and "risk_tier" in alert_df else 0
    high_count     = int((alert_df["risk_tier"] == "HIGH").sum())     if not alert_df.empty and "risk_tier" in alert_df else 0
    avg_prob       = float(alert_df["probability"].mean())            if not alert_df.empty and "probability" in alert_df else 0.0
    total_exposure = float(alert_df["amount"].sum())                  if not alert_df.empty and "amount" in alert_df else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total alerts",    f"{total_alerts:,}",        help="All fraud alerts in the log")
    c2.metric("Critical",        f"{critical_count:,}",      help="Auto-blocked — prob >= 0.70")
    c3.metric("High risk",       f"{high_count:,}",          help="Manual review — prob 0.40 to 0.70")
    c4.metric("Avg probability", f"{avg_prob:.4f}",          help="Mean fraud probability across all alerts")
    c5.metric("Exposure",        f"${total_exposure:,.2f}",  help="Sum of alerted transaction amounts")

    _divider()

    if alert_df.empty:
        st.markdown(
            '<div class="infobox">'
            'No alert data yet. The Overview page populates once the API is running and '
            'the transaction simulator has sent some traffic.<br><br>'
            'Start the simulator in a new terminal tab:<br>'
            '<code>python simulation/real_time_transactions.py --tps 2 --duration 120</code>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    if "timestamp" in alert_df.columns and alert_df["timestamp"].notna().any():
        _section("Alert volume per minute")
        ts = (
            alert_df
            .dropna(subset=["timestamp"])
            .set_index("timestamp")
            .resample("1min")
            .size()
            .rename("Alerts")
        )
        if len(ts) > 1:
            st.area_chart(ts, color="#0969da")
        else:
            st.caption("Keep the simulator running to build a time-series.")

    _divider()
    _section("Distribution by risk tier")

    col_l, col_r = st.columns(2)

    with col_l:
        if "risk_tier" in alert_df.columns and MPL_OK:
            _mpl_style()
            order  = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
            counts = alert_df["risk_tier"].value_counts().reindex(order).dropna().astype(int)
            colors = [TIER_COLOR.get(t, "#57606a") for t in counts.index]
            fig, ax = plt.subplots(figsize=(5, 2.8))
            ax.barh(counts.index, counts.values, color=colors, height=0.5)
            ax.set_xlabel("Alert count", fontsize=8)
            for i, (idx, val) in enumerate(zip(counts.index, counts.values)):
                ax.text(val + max(counts.values) * 0.02, i,
                        str(val), va="center", fontsize=8, color="#57606a")
            plt.tight_layout(pad=1.0)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with col_r:
        if "risk_tier" in alert_df.columns and "amount" in alert_df.columns and MPL_OK:
            _mpl_style()
            order        = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
            tier_amounts = (
                alert_df.groupby("risk_tier")["amount"]
                .mean()
                .reindex(order)
                .dropna()
                .round(2)
            )
            colors = [TIER_COLOR.get(t, "#57606a") for t in tier_amounts.index]
            fig, ax = plt.subplots(figsize=(5, 2.8))
            ax.barh(tier_amounts.index, tier_amounts.values, color=colors, height=0.5)
            ax.set_xlabel("Average amount (USD)", fontsize=8)
            plt.tight_layout(pad=1.0)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ── Page 2 — Live Prediction ───────────────────────────────────────────────────

def _page_live_prediction() -> None:
    st.title("Live Prediction")
    st.caption("Score a single transaction against the live model.")

    if "features" not in st.session_state:
        st.session_state["features"] = dict(_EXAMPLE_LEGIT)

    b1, b2, b3, _ = st.columns([1.3, 1.3, 1.3, 5])
    if b1.button("Legit example"):
        st.session_state["features"] = dict(_EXAMPLE_LEGIT)
        st.rerun()
    if b2.button("Fraud example"):
        st.session_state["features"] = dict(_EXAMPLE_FRAUD)
        st.rerun()
    if b3.button("Randomise"):
        rng = np.random.default_rng()
        st.session_state["features"] = {
            **{f"V{i}": round(float(rng.normal(0, 2)), 4) for i in range(1, 29)},
            "Amount": round(float(rng.exponential(80)), 2),
            "Time":   round(float(rng.uniform(0, 172_800)), 1),
        }
        st.rerun()

    st.write("")
    feats = st.session_state["features"]

    with st.expander("Edit PCA features  V1 to V28", expanded=False):
        st.caption(
            "V14, V12, and V10 are the three strongest fraud predictors in this dataset. "
            "V1 through V28 are PCA-transformed components."
        )
        cols_v = st.columns(7)
        v_vals: dict = {}
        for idx in range(1, 29):
            k = f"V{idx}"
            v_vals[k] = cols_v[(idx - 1) % 7].number_input(
                k,
                value=float(feats.get(k, 0.0)),
                step=0.01,
                format="%.4f",
                help=FEATURE_NOTES.get(k),
                key=f"inp_{k}",
            )

    ca, ct, cex, _ = st.columns([2, 2, 2, 3])
    with ca:
        amount = st.number_input(
            "Amount (USD)", min_value=0.0,
            value=float(feats.get("Amount", 0.0)),
            step=1.0, format="%.2f",
            help="Raw transaction amount. Scaled internally by the API.",
            key="inp_Amount",
        )
    with ct:
        time_val = st.number_input(
            "Time (seconds)", min_value=0.0,
            value=float(feats.get("Time", 0.0)),
            step=1.0, format="%.1f",
            help="Seconds since the first transaction in the dataset (0 to 172800).",
            key="inp_Time",
        )
    with cex:
        st.write("")
        explain_mode = st.checkbox(
            "Include SHAP explanation",
            value=False,
            help="Adds per-feature SHAP contributions to the response",
        )

    current = {**v_vals, "Amount": amount, "Time": time_val}
    st.session_state["features"] = current

    st.write("")

    if not st.button("Score this transaction", type="primary"):
        return

    health = _api_health()
    if not health.get("model_loaded", False):
        st.error(
            "The inference API is not running or the model is not loaded.  "
            "Start it with:  uvicorn api.main:app --host 0.0.0.0 --port 8000"
        )
        return

    with st.spinner("Scoring..."):
        result = _api_predict(current, explain=explain_mode)

    if result is None:
        st.error("No response from the API. Confirm it is running on port 8000.")
        return

    is_fraud = result.get("prediction") == "fraud"
    prob     = float(result.get("probability", 0.0))
    tier     = result.get("risk_tier", "LOW")
    message  = result.get("message", "")
    bar_col  = TIER_COLOR.get(tier, "#1a7f37")
    pct      = round(prob * 100, 2)

    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        score_class = "verdict-score-fraud" if is_fraud else "verdict-score-legit"
        verdict_class = "verdict-fraud" if is_fraud else "verdict-legit"
        decision = "Fraud detected" if is_fraud else "Legitimate transaction"
        st.markdown(
            f'<div class="{verdict_class}">'
            f'<div class="verdict-eyebrow">Model decision</div>'
            f'<div class="verdict-title">{decision}</div>'
            f'<div class="{score_class}">{prob:.4f}</div>'
            f'<div style="margin-top:6px">{_tier_badge(tier)}</div>'
            f'<div style="font-size:0.72rem;color:#57606a;margin-top:6px;">{message}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("**Probability gauge**")
        st.markdown(
            f'<div class="gauge-track">'
            f'<div style="background:{bar_col};width:{max(pct,0.3):.2f}%;'
            f'height:10px;border-radius:6px;transition:width 0.4s;"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:0.7rem;color:#57606a;">'
            f'<span>0.00</span>'
            f'<span style="font-family:monospace;color:{bar_col};'
            f'font-weight:600;font-size:0.82rem;">{prob:.4f}</span>'
            f'<span>1.00</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.caption("Threshold guide")
        st.dataframe(
            pd.DataFrame([
                {"Tier": "Low",      "Range": "below 0.15",   "Action": "Allow"},
                {"Tier": "Medium",   "Range": "0.15 to 0.40", "Action": "Soft review"},
                {"Tier": "High",     "Range": "0.40 to 0.70", "Action": "Manual review"},
                {"Tier": "Critical", "Range": "0.70 and above", "Action": "Auto-block"},
            ]),
            use_container_width=True,
            hide_index=True,
        )

    _divider()
    _section("API response")
    col_j, _ = st.columns([2, 3])
    with col_j:
        st.json({
            "prediction":     result.get("prediction"),
            "probability":    round(prob, 6),
            "risk_tier":      tier,
            "threshold_used": result.get("threshold_used"),
            "message":        message,
        })

    _divider()
    _section("Features submitted to model")
    st.caption(
        "RobustScaler is applied to Amount and Time internally inside the API. "
        "V1 through V28 pass through unchanged."
    )
    st.dataframe(
        pd.DataFrame([
            {"Feature": k, "Value": round(v, 4), "Note": FEATURE_NOTES.get(k, "")}
            for k, v in current.items()
        ]),
        use_container_width=True,
        hide_index=True,
    )

    explanation = result.get("explanation")
    if explanation and MPL_OK:
        _divider()
        _section("SHAP feature contributions")
        st.caption(
            f"Red bars push toward fraud, green bars push toward legitimate.  "
            f"Base value (model prior): {explanation.get('base_value', 0):.4f}"
        )
        top_feats = explanation.get("top_features", [])
        if top_feats:
            _mpl_style()
            names  = [f["feature"]    for f in top_feats]
            values = [f["shap_value"] for f in top_feats]
            colors = [TIER_COLOR["CRITICAL"] if v > 0 else TIER_COLOR["LOW"] for v in values]
            fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.42)))
            ax.barh(range(len(names)), values[::-1], color=colors[::-1], height=0.5)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names[::-1], fontsize=8)
            ax.axvline(0, color="#d0d7de", linewidth=0.8, linestyle="--")
            ax.set_xlabel("SHAP contribution to fraud probability", fontsize=8)
            plt.tight_layout(pad=1.0)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if text := explanation.get("explanation_text"):
            for line in text.split("\n"):
                if line.strip():
                    st.caption(line)

        with st.expander("Raw SHAP data"):
            st.json(explanation)

    elif explain_mode:
        st.info("SHAP unavailable. Install with:  pip install shap")


# ── Page 3 — Model Analysis ────────────────────────────────────────────────────

def _page_model_analysis() -> None:
    st.title("Model Analysis")
    st.caption(
        "Evaluation on the held-out test set — 20% of data, "
        "never seen during training or threshold calibration."
    )

    meta = _load_model_metadata()
    if not meta:
        st.markdown(
            '<div class="infobox">No model metadata found. '
            'Run <code>python main.py</code> to train.</div>',
            unsafe_allow_html=True,
        )
        return

    _section("Validation metrics")

    specs = [
        ("val_pr_auc",    "PR-AUC",     "Primary metric for imbalanced datasets. Area under Precision-Recall curve."),
        ("val_roc_auc",   "ROC-AUC",    "Area under the ROC curve."),
        ("val_precision", "Precision",   "TP / (TP + FP)  —  fraction of fraud alerts that were real fraud."),
        ("val_recall",    "Recall",      "TP / (TP + FN)  —  fraction of actual fraud cases caught."),
        ("val_f1",        "F1 Score",    "Harmonic mean of Precision and Recall."),
    ]
    cols = st.columns(len(specs))
    for col, (key, label, tip) in zip(cols, specs):
        val = meta.get(key)
        col.metric(label, f"{float(val):.4f}" if val is not None else "—", help=tip)

    _divider()
    _section("Confusion matrix and feature importance")

    col_l, col_r = st.columns(2)
    with col_l:
        candidates = [
            MODEL_DIR / "xgboost_confusion_matrix.png",
            MODEL_DIR / "random_forest_confusion_matrix.png",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path:
            st.image(str(path), use_container_width=True)
            st.caption(
                "Rows = actual class, columns = predicted. "
                "We optimise for high recall at the cost of some precision — "
                "missing a fraud costs far more than a false alert."
            )
        else:
            st.markdown(
                '<div class="infobox">Run <code>python main.py</code> to generate charts.</div>',
                unsafe_allow_html=True,
            )

    with col_r:
        candidates = [
            MODEL_DIR / "xgboost_feature_importance.png",
            MODEL_DIR / "random_forest_feature_importance.png",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path:
            st.image(str(path), use_container_width=True)
            st.caption(
                "V14, V12, and V10 are consistently the strongest fraud signals. "
                "Engineered features log_amount and V12_V14 also rank highly."
            )
        else:
            st.markdown(
                '<div class="infobox">Run <code>python main.py</code> to generate charts.</div>',
                unsafe_allow_html=True,
            )

    _divider()
    _section("ROC and Precision-Recall curves")

    col_roc, col_pr = st.columns(2)
    with col_roc:
        p = REPORT_DIR / "roc_curves.png"
        if p.exists():
            st.image(str(p), use_container_width=True)
        else:
            st.markdown(
                '<div class="infobox">Run <code>python main.py</code> to generate plots.</div>',
                unsafe_allow_html=True,
            )

    with col_pr:
        p = REPORT_DIR / "pr_curves.png"
        if p.exists():
            st.image(str(p), use_container_width=True)
            st.caption(
                "The correct primary chart for imbalanced datasets. "
                "A random classifier sits at precision = 0.0017 (the base fraud rate)."
            )
        else:
            st.markdown(
                '<div class="infobox">Run <code>python main.py</code> to generate plots.</div>',
                unsafe_allow_html=True,
            )

    shap_s = REPORT_DIR / "shap_summary.png"
    shap_b = REPORT_DIR / "shap_bar.png"
    if shap_s.exists() or shap_b.exists():
        _divider()
        _section("SHAP global explainability")
        st.caption(
            "SHAP values show each feature's average contribution to fraud predictions "
            "across the entire test set."
        )
        cs, cb = st.columns(2)
        with cs:
            if shap_s.exists():
                st.image(str(shap_s), use_container_width=True)
                st.caption(
                    "Beeswarm — each dot is one transaction. "
                    "Position on the x-axis = SHAP impact on fraud probability."
                )
        with cb:
            if shap_b.exists():
                st.image(str(shap_b), use_container_width=True)
                st.caption("Mean absolute SHAP value per feature across all test predictions.")

    results_df = _load_model_results()
    if not results_df.empty:
        _divider()
        _section("Model comparison — test set")
        st.caption("All models evaluated at threshold 0.40 on the held-out test set.")
        st.dataframe(
            results_df.sort_values("PR_AUC", ascending=False).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Training hyperparameters"):
        params = meta.get("params", {})
        if params:
            st.dataframe(
                pd.DataFrame(
                    [{"Parameter": k, "Value": v} for k, v in sorted(params.items())]
                ),
                use_container_width=True,
                hide_index=True,
            )
        tc, sc = st.columns(2)
        if t := meta.get("train_time_seconds"):
            tc.metric("Training time", f"{float(t):.1f} s")
        if n := meta.get("train_size"):
            sc.metric("Training rows", f"{int(n):,}")


# ── Page 4 — Alert Feed ────────────────────────────────────────────────────────

def _page_alert_feed() -> None:
    st.title("Alert Feed")
    st.caption("Live fraud alerts written to the JSONL log by the alert system.")

    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
    with ctrl1:
        min_tier = st.selectbox("Minimum severity", ["All", "Medium", "High", "Critical"])
    with ctrl2:
        n_show = st.slider("Max alerts to display", 10, 200, 50)
    with ctrl3:
        st.write("")
        if st.button("Refresh"):
            st.cache_data.clear()

    alert_df = _load_alert_log(n_show * 5)

    if alert_df.empty:
        st.markdown(
            '<div class="infobox">'
            'No alerts yet. Run the simulator to generate traffic.<br><br>'
            '<code>python simulation/real_time_transactions.py --tps 2 --duration 120</code>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    _rank    = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    min_rank = _rank.get(min_tier.upper(), -1) if min_tier != "All" else -1
    if min_rank >= 0 and "risk_tier" in alert_df.columns:
        alert_df = alert_df[
            alert_df["risk_tier"].map(lambda t: _rank.get(t, 0)) >= min_rank
        ]

    display = alert_df.tail(n_show).iloc[::-1]

    if display.empty:
        st.info(f"No alerts at or above {min_tier} severity.")
        return

    for _, row in display.iterrows():
        tier   = str(row.get("risk_tier", "LOW")).upper()
        prob   = float(row.get("probability", 0.0))
        amount = float(row.get("amount", 0.0))
        txn_id = str(row.get("transaction_id", "UNKNOWN"))
        ts     = str(row.get("timestamp", ""))[:19]
        action = str(row.get("action", ""))
        row_css = tier.lower()

        st.markdown(
            f'<div class="alert-item {row_css}">'
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<span class="mono">{txn_id}</span>'
            f'&nbsp;{_tier_badge(tier)}'
            f'</div>'
            f'<div class="small-muted">'
            f'{ts}&nbsp;&nbsp;·&nbsp;&nbsp;'
            f'${amount:,.2f}&nbsp;&nbsp;·&nbsp;&nbsp;'
            f'score&nbsp;{prob:.4f}&nbsp;&nbsp;·&nbsp;&nbsp;'
            f'{action}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _divider()
    with st.expander("Raw data table and export"):
        clean = display.reset_index(drop=True)
        st.dataframe(clean, use_container_width=True)
        st.download_button(
            "Download as CSV",
            data=clean.to_csv(index=False).encode(),
            file_name="fraud_alerts_export.csv",
            mime="text/csv",
        )


# ── Page 5 — Batch Scoring ─────────────────────────────────────────────────────

def _page_batch_scoring() -> None:
    st.title("Batch Scoring")
    st.caption(
        "Upload a CSV and score every row against the live model. "
        "Required columns: V1 through V28, Amount, Time. "
        "Maximum 30,000 rows per upload."
    )

    uploaded = st.file_uploader("Upload transactions CSV", type="csv")
    if uploaded is None:
        st.markdown(
            '<div class="infobox">'
            'Awaiting a CSV file with columns <code>V1</code> through <code>V28</code>, '
            '<code>Amount</code>, and <code>Time</code>.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not parse file: {exc}")
        return

    st.success(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")

    required = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    max_rows = min(len(df), 30_000)
    if len(df) > max_rows:
        st.warning(f"Only the first {max_rows:,} rows will be scored.")

    if not st.button("Score all transactions", type="primary"):
        return

    health = _api_health()
    if not health.get("model_loaded", False):
        st.error(
            "API model is not loaded. "
            "Start with:  uvicorn api.main:app --host 0.0.0.0 --port 8000"
        )
        return

    bar      = st.progress(0.0, text="Scoring transactions...")
    results: list[dict] = []

    for idx, (_, row) in enumerate(df.head(max_rows).iterrows()):
        payload = {col: float(row[col]) for col in required}
        resp    = _api_predict(payload)
        if resp:
            results.append({
                "prediction":  resp["prediction"],
                "probability": resp["probability"],
                "risk_tier":   resp["risk_tier"],
            })
        else:
            results.append({
                "prediction": "error",
                "probability": 0.0,
                "risk_tier":   "UNKNOWN",
            })
        bar.progress((idx + 1) / max_rows, text=f"Scored {idx + 1:,} of {max_rows:,}")

    bar.empty()
    res_df = pd.DataFrame(results)

    total       = len(res_df)
    fraud_count = int((res_df["prediction"] == "fraud").sum())
    fraud_rate  = fraud_count / total if total > 0 else 0.0

    _divider()
    _section("Summary")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total scored",  f"{total:,}")
    k2.metric("Fraud",         f"{fraud_count:,}")
    k3.metric("Legitimate",    f"{total - fraud_count:,}")
    k4.metric("Fraud rate",    f"{fraud_rate:.2%}")

    _divider()
    _section("Score distribution")

    col_t, col_p = st.columns(2)

    with col_t:
        st.caption("Risk tier breakdown")
        if MPL_OK:
            _mpl_style()
            order       = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]
            tier_counts = (
                res_df["risk_tier"].value_counts().reindex(order).dropna().astype(int)
            )
            colors = [TIER_COLOR.get(t, "#57606a") for t in tier_counts.index]
            fig, ax = plt.subplots(figsize=(5, 2.8))
            ax.bar(tier_counts.index, tier_counts.values, color=colors, width=0.45)
            ax.set_ylabel("Count", fontsize=8)
            plt.tight_layout(pad=1.0)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with col_p:
        st.caption("Fraud probability distribution")
        probs = res_df["probability"]
        if MPL_OK and len(probs) > 0:
            _mpl_style()
            fig, ax = plt.subplots(figsize=(5, 2.8))
            ax.hist(probs, bins=40, color="#0969da", alpha=0.8)
            ax.axvline(
                0.40, color=TIER_COLOR["HIGH"],
                linewidth=1.2, linestyle="--", label="Threshold 0.40"
            )
            ax.set_xlabel("Fraud probability", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.legend(fontsize=7)
            plt.tight_layout(pad=1.0)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    _divider()
    _section("Full scored results")

    final_df = pd.concat(
        [df.head(max_rows).reset_index(drop=True), res_df.reset_index(drop=True)],
        axis=1,
    )
    st.dataframe(final_df, use_container_width=True)
    st.download_button(
        "Download scored results (CSV)",
        data=final_df.to_csv(index=False).encode(),
        file_name="scored_transactions.csv",
        mime="text/csv",
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    page = _render_sidebar()
    dispatch = {
        "Overview":        _page_overview,
        "Live Prediction": _page_live_prediction,
        "Model Analysis":  _page_model_analysis,
        "Alert Feed":      _page_alert_feed,
        "Batch Scoring":   _page_batch_scoring,
    }
    fn = dispatch.get(page)
    if fn:
        fn()


if __name__ == "__main__":
    main()
