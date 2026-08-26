"""
app.py
======
AI-Powered Network Intrusion Detection System — LIVE DASHBOARD

Active detection architecture (Phase 2):
    LIVE PACKETS -> live_capture.py -> flow_extractor.py
    -> 49 FEATURES -> XGBoost + Autoencoder -> Fusion Engine
    -> DASHBOARD (auto-updates every 1.5 seconds)

Run:
    python app.py

Prerequisites:
  1. python train_xgboost_ae.py   (once, to save models/)
  2. Npcap installed              (https://npcap.com/)
  3. Run as Administrator         (required for raw packet capture on Windows)

Configuration:
  Set CAPTURE_IFACE below if you want to hard-code an interface name.
  Leave as None to be prompted at startup.

NOTE: Random Forest has been removed from the active pipeline.
      The dashboard no longer contains stream-position/replay controls.
      The dashboard automatically follows the latest live data.
"""

import os
import sys
import logging
import warnings

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  <- edit here if needed
# ─────────────────────────────────────────────────────────────────────────────
CAPTURE_IFACE : str = "Wi-Fi"   # e.g. "Wi-Fi" or "Ethernet" or None for prompt
DASHBOARD_PORT: int = 8051
WINDOW_SIZE   : int = 50     # number of recent flows shown in graphs / cards
# ─────────────────────────────────────────────────────────────────────────────

# ── Verify required model artifacts exist ─────────────────────────────────────
from config import MODELS_DIR, AE_MODEL_PATH, SCALER_PATH, AE_ARTIFACTS_PATH, XGB_MODEL_PATH, XGB_ARTIFACTS_PATH
_REQUIRED = [
    (AE_MODEL_PATH,     "Run  python train_xgboost_ae.py  first."),
    (SCALER_PATH,       "Run  python train_xgboost_ae.py  first."),
    (AE_ARTIFACTS_PATH, "Run  python train_xgboost_ae.py  first."),
    (XGB_MODEL_PATH,    "Run  python train_xgboost_ae.py  first."),
    (XGB_ARTIFACTS_PATH,"Run  python train_xgboost_ae.py  first."),
]
for _f, _hint in _REQUIRED:
    if not os.path.exists(_f):
        print(
            f"\n[ERROR] Missing artifact: models/{_f}\n"
            f"  {_hint}\n"
        )
        sys.exit(1)

# ── Load inference engine (XGBoost + AE + Fusion) ────────────────────────────
from inference import engine as _inference_engine
_inference_engine.load()

# ── Interface selection ───────────────────────────────────────────────────────
from live_capture import (
    select_interface_interactive,
    start_capture,
    stop_capture,
    get_flows_snapshot,
    capture_status,
)

if CAPTURE_IFACE is None:
    CAPTURE_IFACE = select_interface_interactive()

# ── Start capture ─────────────────────────────────────────────────────────────
start_capture(iface=CAPTURE_IFACE)
logger.info(
    "Live capture running on '%s'. Dashboard starting ...",
    CAPTURE_IFACE or "default",
)

# ── Dash imports ──────────────────────────────────────────────────────────────
import dash
from dash import Dash, html, dcc, dash_table, Input, Output
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# ── App ───────────────────────────────────────────────────────────────────────
app = Dash(__name__)
app.title = "AI-Powered Network Intrusion Detection System"


# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div([

    html.H2("AI-Powered Network Intrusion Detection System — Live Dashboard"),

    # Live capture status bar
    html.Div(
        id="capture-status-bar",
        style={
            "padding"      : "6px 12px",
            "marginBottom" : "6px",
            "background"   : "#f0f8ff",
            "border"       : "1px solid #b0d0f0",
            "borderRadius" : "4px",
            "fontSize"     : "13px",
            "color"        : "#336",
        }
    ),

    # Summary cards row
    html.Div(
        id="summary-cards",
        style={
            "display": "flex",
            "gap"    : "16px",
            "flexWrap": "wrap",
            "marginBottom": "10px",
        }
    ),

    # Traffic + alerts time series
    dcc.Graph(id="time-series-agg"),

    # Score vs packet rate scatter
    dcc.Graph(id="scatter-score"),

    html.H4("Recent Live Flows"),

    dash_table.DataTable(
        id="recent-table",

        columns=[
            {"name": c, "id": c}
            for c in [
                "timestamp",
                "src_ip",
                "dst_ip",
                "src_port",
                "dst_port",
                "protocol",
                "pkt_count",
                "byte_count",
                "duration",
                "true_label",
                "xgb_prob",
                "ae_score",
                "fusion_score",
                "fusion_mode",
            ]
        ],

        data=[],
        page_size=15,

        style_table={"overflowX": "auto"},

        style_data_conditional=[
            {
                "if"            : {"filter_query": "{xgb_prob} >= 0.5"},
                "backgroundColor": "#fff0f0",
                "color"         : "#900",
            },
            {
                "if"            : {"filter_query": "{fusion_pred} = 1"},
                "backgroundColor": "#ffeaea",
                "color"         : "#800",
            },
        ],
    ),

    # Auto-refresh interval — dashboard follows latest live data automatically
    dcc.Interval(
        id="interval",
        interval=1500,   # ms
        n_intervals=0,
    ),

], style={"padding": "12px"})


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: build a DataFrame from the trailing WINDOW_SIZE live flows
# The dashboard always auto-follows the newest data (no stream position needed).
# ─────────────────────────────────────────────────────────────────────────────

def _get_window() -> pd.DataFrame:
    flows = get_flows_snapshot()
    if not flows:
        return pd.DataFrame()

    # Always take the most recent WINDOW_SIZE flows
    subset = flows[-WINDOW_SIZE:]
    if not subset:
        return pd.DataFrame()

    df = pd.DataFrame(subset)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK — Dashboard auto-update
# Triggered by the interval component every 1.5 seconds.
# No stream-position slider or play/pause controls.
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("recent-table",    "data"),
    Output("time-series-agg", "figure"),
    Output("scatter-score",   "figure"),
    Output("summary-cards",   "children"),
    Output("capture-status-bar", "children"),
    Input("interval",         "n_intervals"),
)
def update_ui(_n_intervals):

    window = _get_window()

    # ── Capture status bar ────────────────────────────────────────────────────
    s = capture_status
    status_text = (
        f"LIVE  |  Interface: {s['iface'] or 'default'}  |  "
        f"Packets: {s['packets']:,}  |  "
        f"IPv4: {s['ipv4_packets']:,}  |  "
        f"Active flows: {s['active_flows']:,}  |  "
        f"Finalized: {s['flows_finalized']:,}  |  "
        f"Inference: {s['flows_to_inference']:,}  |  "
        f"Dashboard flows: {s['flows']:,}  |  "
        f"Schema errors: {s['validate_failures']}  |  "
        f"Errors: {s['errors']}"
        + (f"  <- {s['last_error']}" if s["errors"] and s["last_error"] else "")
    )

    # ── Summary cards ─────────────────────────────────────────────────────────
    total       = len(window)
    xgb_alerts  = int((window["xgb_pred"]   == 1).sum()) if total else 0
    ae_alerts   = int((window["ae_pred"]     == 1).sum()) if total else 0
    fusion_alerts = int((window["fusion_pred"] == 1).sum()) if total else 0
    fusion_mode = window["fusion_mode"].iloc[-1] if total else "N/A"

    _card_style_base = {
        "padding"     : "10px 16px",
        "border"      : "1px solid #ddd",
        "borderRadius": "6px",
        "minWidth"    : "130px",
    }

    cards = [
        html.Div([
            html.H4("Window", style={"margin": "0 0 4px 0", "fontSize": "13px", "color": "#555"}),
            html.P(str(total), style={"margin": 0, "fontSize": "22px", "fontWeight": "bold"}),
        ], style=_card_style_base),

        html.Div([
            html.H4("True Attacks", style={"margin": "0 0 4px 0", "fontSize": "13px", "color": "#555"}),
            html.P("N/A - Live Traffic", style={"margin": 0, "fontSize": "13px", "color": "#888"}),
        ], style={**_card_style_base, "border": "1px solid #f88", "backgroundColor": "#ffeeee"}),

        html.Div([
            html.H4("XGBoost Alerts", style={"margin": "0 0 4px 0", "fontSize": "13px", "color": "#555"}),
            html.P(str(xgb_alerts), style={"margin": 0, "fontSize": "22px", "fontWeight": "bold", "color": "#c04"}),
        ], style={**_card_style_base, "border": "1px solid #c0a0f0"}),

        html.Div([
            html.H4("AE Alerts", style={"margin": "0 0 4px 0", "fontSize": "13px", "color": "#555"}),
            html.P(str(ae_alerts), style={"margin": 0, "fontSize": "22px", "fontWeight": "bold", "color": "#048"}),
        ], style={**_card_style_base, "border": "1px solid #8cf8d8"}),

        html.Div([
            html.H4("Fusion Alerts", style={"margin": "0 0 4px 0", "fontSize": "13px", "color": "#555"}),
            html.P(str(fusion_alerts), style={"margin": 0, "fontSize": "22px", "fontWeight": "bold", "color": "#840"}),
        ], style={**_card_style_base, "border": "1px solid #f0a030", "backgroundColor": "#fffbf0"}),

        html.Div([
            html.H4("Fusion Mode", style={"margin": "0 0 4px 0", "fontSize": "13px", "color": "#555"}),
            html.P(
                fusion_mode.upper() if total else "N/A",
                style={"margin": 0, "fontSize": "14px", "fontWeight": "bold",
                       "color": "#060" if fusion_mode == "full" else "#840"},
            ),
            html.P(
                "(GNN/Temporal pending)" if fusion_mode == "partial" else "",
                style={"margin": 0, "fontSize": "11px", "color": "#888"},
            ),
        ], style={**_card_style_base, "border": "1px solid #aaa"}),
    ]

    # ── Time-series traffic + alerts graph ────────────────────────────────────
    if window.empty:
        fig_ts = go.Figure()
        fig_ts.update_layout(title="Traffic count (recent window)")
    else:
        agg = (
            window
            .groupby(window["timestamp"].dt.floor("s"))
            .agg(
                total        =("timestamp", "count"),
                xgb_alerts   =("xgb_pred",   lambda s: (s == 1).sum()),
                ae_alerts    =("ae_pred",     lambda s: (s == 1).sum()),
                fusion_alerts=("fusion_pred", lambda s: (s == 1).sum()),
            )
            .reset_index()
        )

        fig_ts = go.Figure()
        if not agg.empty:
            fig_ts.add_trace(go.Bar(
                x=agg["timestamp"], y=agg["total"],
                name="Total Traffic", marker_color="#aec7e8",
            ))
            fig_ts.add_trace(go.Bar(
                x=agg["timestamp"], y=agg["xgb_alerts"],
                name="XGBoost Alerts", marker_color="#d62728",
            ))
            fig_ts.add_trace(go.Bar(
                x=agg["timestamp"], y=agg["ae_alerts"],
                name="AE Alerts", marker_color="#9467bd",
            ))
            fig_ts.add_trace(go.Bar(
                x=agg["timestamp"], y=agg["fusion_alerts"],
                name="Fusion Alerts", marker_color="#ff7f0e",
            ))
            fig_ts.update_layout(
                barmode="overlay",
                title="Traffic count — recent window (auto-following live data)",
                legend=dict(orientation="h", y=-0.15),
            )

    # ── Fusion score vs Packet Rate scatter ───────────────────────────────────
    if window.empty:
        fig_sc = go.Figure()
        fig_sc.update_layout(title="Fusion score vs Packet rate (recent)")
    else:
        window_plot = window.copy()
        window_plot["xgb_label"] = window_plot["xgb_pred"].map(
            {0: "Normal (XGB)", 1: "Attack (XGB)"}
        ).fillna("Unknown")

        # Only include columns that exist (ae_score may not always be present in edge cases)
        hover_cols = [c for c in ["src_ip", "dst_ip", "xgb_prob", "ae_score", "fusion_mode"]
                      if c in window_plot.columns]

        fig_sc = px.scatter(
            window_plot,
            x="pkt_rate",
            y="fusion_score",
            color="xgb_label",
            color_discrete_map={
                "Normal (XGB)": "#1f77b4",
                "Attack (XGB)": "#d62728",
            },
            hover_data=hover_cols,
        )
        fig_sc.add_hline(
            y=_inference_engine._fusion.threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Fusion threshold ({_inference_engine._fusion.threshold})",
        )
        fig_sc.update_layout(
            title="Fusion score vs Packet rate (recent) — coloured by XGBoost prediction"
        )

    # ── Recent events table ───────────────────────────────────────────────────
    recent_cols = [
        "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
        "protocol", "pkt_count", "byte_count", "duration",
        "true_label", "xgb_prob", "ae_score", "fusion_score", "fusion_mode",
    ]

    if window.empty:
        recent_data = []
    else:
        present_cols = [c for c in recent_cols if c in window.columns]
        recent_data = (
            window
            .sort_values("timestamp", ascending=False)
            .head(20)[present_cols]
            .to_dict("records")
        )

    return (
        recent_data,
        fig_ts,
        fig_sc,
        cards,
        status_text,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import atexit
    atexit.register(stop_capture)

    print(f"\nDashboard -> http://127.0.0.1:{DASHBOARD_PORT}\n")

    app.run(
        debug=False,
        port=DASHBOARD_PORT,
        use_reloader=False,     # MUST be False — reloader kills the capture thread
    )
