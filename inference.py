"""
inference.py
============
AI-Powered Network Traffic Analyzer — INFERENCE ENGINE

Active detection architecture (Full):
    LIVE FLOW -> 49 FEATURES -> XGBoost + Autoencoder + GNN -> Fusion Engine -> Dashboard

This module:
  - Loads XGBoost, Autoencoder, Scaler, GNN artifacts from models/
  - Validates the canonical 49-feature schema on every flow
  - Runs XGBoost, AE, and GNN (OnlineGraphStream) inference
  - Extracts enriched behavioral features (IAT, TCP flags, payload stats)
  - Produces normalized scores for the Dynamic Fusion Engine
  - Returns a structured result dict with full fusion output

Feature schema is imported from feature_schema.py — single canonical source.
Missing features raise ValueError — they are NEVER silently fabricated.

Usage:
    from inference import engine
    engine.load()        # load AE + Scaler + XGBoost + GNN
    result = engine.run_inference(feature_dict)
"""

import os
import json
import time
import logging
import datetime

import numpy as np

from feature_schema import FEATURE_COLS
from fusion.fusion_engine import FusionEngine, load_fusion_threshold
from features.enriched_features import extract_from_dataset_record, ENRICHED_FEATURE_KEYS

logger = logging.getLogger(__name__)

_MODELS_DIR    = os.path.join(os.path.dirname(__file__), "models")
_AE_PATH       = os.path.join(_MODELS_DIR, "ae_model.keras")
_SCALER_PATH   = os.path.join(_MODELS_DIR, "scaler.pkl")
_ART_PATH      = os.path.join(_MODELS_DIR, "artifacts.json")    # AE threshold + mse_max
_XGB_PATH      = os.path.join(_MODELS_DIR, "xgb_model.pkl")
_XGB_ART_PATH  = os.path.join(_MODELS_DIR, "xgb_artifacts.json")
_GNN_MODEL_PATH   = os.path.join(os.path.dirname(__file__), "dynamic_temporal_gnn.pt")
_GNN_SCALERS_PATH = os.path.join(os.path.dirname(__file__), "gnn_scalers.pkl")


class InferenceEngine:
    """
    Active inference engine: XGBoost + Autoencoder + Fusion.

    XGBoost is the primary supervised classifier.
    Autoencoder provides unsupervised anomaly detection.
    Fusion engine combines both into a single score (partial mode until GNN/Temporal).

    Random Forest has been removed from the active pipeline.
    """

    def __init__(self):
        # ── Autoencoder ──────────────────────────────────────────────────────
        self._ae            = None
        self._scaler        = None
        self._ae_threshold  = None   # from artifacts.json — used for ae_pred
        self._ae_mse_max    = None   # from xgb_artifacts.json — leakage-free normalization

        # ── XGBoost ──────────────────────────────────────────────────────────
        self._xgb           = None
        self._xgb_attack_idx = None

        # ── GNN (OnlineGraphStream) ───────────────────────────────────────────
        self._gnn_stream    = None   # OnlineGraphStream instance
        self._gnn_loaded    = False

        # ── Fusion ───────────────────────────────────────────────────────────
        self._fusion        = None

        self._loaded        = False
        self._xgb_loaded    = False

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self):
        """
        Load AE + Scaler + XGBoost + GNN artifacts from disk.

        Required files:
            models/ae_model.keras
            models/scaler.pkl
            models/artifacts.json     (AE threshold)
            models/xgb_model.pkl
            models/xgb_artifacts.json (xgb_ae_mse_max for leakage-free AE normalization)

        Optional (GNN branch — graceful degradation if missing):
            dynamic_temporal_gnn.pt
            gnn_scalers.pkl

        Raises FileNotFoundError if any required artifact is missing.
        """
        import joblib
        import tensorflow as tf

        required = [
            (_AE_PATH,      "ae_model.keras"),
            (_SCALER_PATH,  "scaler.pkl"),
            (_ART_PATH,     "artifacts.json"),
            (_XGB_PATH,     "xgb_model.pkl"),
            (_XGB_ART_PATH, "xgb_artifacts.json"),
        ]
        for path, label in required:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required artifact not found: {path}\n"
                    f"  ({label})\n"
                    "Run  python train_xgboost_ae.py  first."
                )

        # ── Autoencoder ──────────────────────────────────────────────────────
        logger.info("Loading AE ...")
        self._ae = tf.keras.models.load_model(_AE_PATH)

        logger.info("Loading Scaler ...")
        self._scaler = joblib.load(_SCALER_PATH)

        with open(_ART_PATH) as f:
            art = json.load(f)

        stored_cols = art.get("feature_cols", [])
        if stored_cols != FEATURE_COLS:
            logger.warning(
                "artifacts.json feature_cols differs from FEATURE_COLS in "
                "feature_schema.py. Using feature_schema.py as authoritative source."
            )

        self._ae_threshold = float(art["ae_threshold"])

        # ── XGBoost ──────────────────────────────────────────────────────────
        logger.info("Loading XGBoost ...")
        self._xgb = joblib.load(_XGB_PATH)

        with open(_XGB_ART_PATH) as f:
            xgb_art = json.load(f)

        stored_cols_xgb = xgb_art.get("feature_cols", [])
        if stored_cols_xgb != FEATURE_COLS:
            logger.warning(
                "xgb_artifacts.json feature_cols differs from FEATURE_COLS. "
                "Using feature_schema.py as authoritative source."
            )

        self._ae_mse_max = float(xgb_art["xgb_ae_mse_max"])

        classes = list(self._xgb.classes_)
        self._xgb_attack_idx = int(classes.index(1))
        self._xgb_loaded = True

        # ── GNN (OnlineGraphStream) — optional, graceful degradation ─────────
        if os.path.exists(_GNN_MODEL_PATH) and os.path.exists(_GNN_SCALERS_PATH):
            try:
                import torch
                from gnn.dynamic_temporal_gnn import DynamicBipartiteTemporalGNN
                from gnn.online_graph_stream import OnlineGraphStream

                logger.info("Loading GNN model from %s ...", _GNN_MODEL_PATH)
                gnn_model = DynamicBipartiteTemporalGNN(
                    host_dim=8, service_dim=8, edge_dim=9, hidden_dim=32
                )
                state_dict = torch.load(
                    _GNN_MODEL_PATH, map_location="cpu", weights_only=False
                )
                gnn_model.load_state_dict(state_dict)
                gnn_model.eval()

                logger.info("Loading GNN scalers from %s ...", _GNN_SCALERS_PATH)
                gnn_scalers = joblib.load(_GNN_SCALERS_PATH)

                self._gnn_stream = OnlineGraphStream(
                    gnn_model=gnn_model,
                    scalers=gnn_scalers,
                    window_size=500,
                    mode="live_capture",
                )
                self._gnn_loaded = True
                logger.info("GNN OnlineGraphStream ready (window_size=500).")
            except Exception as exc:
                logger.warning(
                    "GNN loading failed — running in partial mode (XGB+AE only): %s", exc
                )
                self._gnn_loaded = False
        else:
            logger.info(
                "GNN artifacts not found at expected paths — "
                "running in partial mode (XGB+AE only)."
            )
            self._gnn_loaded = False

        # ── Fusion engine ────────────────────────────────────────────────────
        threshold = load_fusion_threshold()
        self._fusion = FusionEngine(threshold=threshold)

        self._loaded = True
        logger.info(
            "Inference engine ready. GNN=%s  "
            "AE threshold=%.6f  ae_mse_max=%.6f (leakage-free)",
            "ACTIVE" if self._gnn_loaded else "DISABLED (partial mode)",
            self._ae_threshold, self._ae_mse_max,
        )

    def validate_features(self, feature_dict: dict) -> bool:
        """
        Validate that all 49 canonical features are present.

        Returns True if all features are present.
        Returns False and logs the missing features otherwise.

        A missing feature is a schema error — it is NEVER silently fabricated.
        A legitimate feature value of 0 is valid.
        """
        if not self._loaded:
            return False
        missing = [c for c in FEATURE_COLS if c not in feature_dict]
        if missing:
            logger.warning(
                "Feature validation FAILED — %d missing feature(s): %s",
                len(missing), missing[:10],
            )
            return False
        return True

    def run_inference(self, feature_dict: dict) -> dict:
        """
        Run XGBoost + AE inference on a single flow and compute the fusion score.

        Parameters
        ----------
        feature_dict : dict
            Must contain all 49 FEATURE_COLS keys (numeric) plus '_*' metadata keys.
            Missing features raise ValueError — they are not fabricated.

        Returns
        -------
        dict with keys:
            timestamp, src_ip, dst_ip, src_port, dst_port, protocol,
            pkt_count, byte_count, duration, pkt_rate,
            true_label,
            xgb_pred, xgb_prob,
            ae_mse, ae_score, ae_pred,
            gnn_score (None — not yet available),
            temporal_score (None — not yet available),
            fusion_score, fusion_pred, fusion_mode,
            inference_latency_ms
        """
        if not self._loaded:
            raise RuntimeError("InferenceEngine.load() has not been called.")

        if not self.validate_features(feature_dict):
            raise ValueError(
                "Feature dict is missing required features. "
                "Missing features are not fabricated — fix the flow extractor."
            )

        t_start = time.perf_counter()

        # ── Build 1-row feature array (unscaled, ordered) ────────────────────
        x_raw = np.array(
            [[feature_dict[c] for c in FEATURE_COLS]],
            dtype=np.float32,
        )
        # Replace any inf / nan with 0 — these are legitimate cleanup steps
        # (inf/nan values are not valid features; 0 is the safe replacement).
        x_raw = np.where(np.isfinite(x_raw), x_raw, 0.0)

        # ── XGBoost inference (unscaled — tree model is scale-invariant) ──────
        xgb_prob = float(
            self._xgb.predict_proba(x_raw)[0, self._xgb_attack_idx]
        )
        xgb_pred = int(self._xgb.predict(x_raw)[0])

        # ── AE inference (scaled input) ───────────────────────────────────────
        x_scaled = self._scaler.transform(x_raw).astype(np.float32)
        recon    = self._ae.predict(x_scaled, verbose=0)
        ae_mse   = float(np.mean(np.square(recon - x_scaled)))
        ae_pred  = int(ae_mse > self._ae_threshold)

        # AE score normalized to [0, 1] using leakage-free xgb_ae_mse_max.
        # Clipped to [0, 1] for fusion. Raw ae_mse is also reported separately.
        ae_score = float(min(ae_mse / (self._ae_mse_max + 1e-9), 1.0))

        # ── Enriched features (IAT, TCP flags, payload stats) ─────────────────
        enriched = extract_from_dataset_record(feature_dict)

        # ── GNN inference via OnlineGraphStream ───────────────────────────────
        gnn_score_val    = None
        temporal_score_val = None
        gnn_latency_ms   = 0.0

        if self._gnn_loaded and self._gnn_stream is not None:
            try:
                # Build a minimal flow_rec for the graph stream
                _flow_rec = {
                    "src_ip"      : feature_dict.get("_src_ip", "0.0.0.0"),
                    "dst_ip"      : feature_dict.get("_dst_ip", "0.0.0.0"),
                    "dst_port"    : feature_dict.get("_dst_port", 0),
                    "protocol"    : feature_dict.get("_protocol", "TCP"),
                    "timestamp"   : feature_dict.get("_timestamp", time.time()),
                    "byte_count"  : feature_dict.get("_byte_count", 0),
                    "pkt_count"   : feature_dict.get("_pkt_count", 0),
                    "duration"    : feature_dict.get("_duration", 0.0),
                    "xgb_prob"    : xgb_prob,
                    "ae_score"    : ae_score,
                    "fusion_score": 0.0,   # placeholder — fusion not yet computed
                    "attack_label": 0,
                }
                self._gnn_stream.ingest_flow(_flow_rec)

                # Only evaluate when we have enough flows for a meaningful graph
                if self._gnn_stream.buffer_size >= 2:
                    from graph.gnn_output import GNNOutputRecord
                    t_gnn = time.perf_counter()
                    probs, temporal_shift, _pyg, _hm, _sm, gnn_lat = \
                        self._gnn_stream.evaluate_realtime_gnn_risk()
                    gnn_latency_ms = gnn_lat

                    if probs is not None and len(probs) > 0:
                        gnn_out = GNNOutputRecord.from_stream_output(
                            flow_id=f"{_flow_rec['src_ip']}:{_flow_rec['dst_port']}",
                            probs=probs,
                            gnn_latency_ms=gnn_lat,
                            stream=self._gnn_stream,
                            edge_idx=-1,
                            temporal_shift=temporal_shift,
                        )
                        gnn_score_val    = gnn_out.gnn_score
                        temporal_score_val = gnn_out.temporal_score
            except Exception as exc:
                logger.debug("GNN inference error (non-fatal): %s", exc)

        # ── Fusion ────────────────────────────────────────────────────────────
        fusion_result = self._fusion.fuse(
            xgb_score=xgb_prob,
            ae_score=ae_score,
            gnn_score=gnn_score_val,
            temporal_score=temporal_score_val,
        )

        t_end = time.perf_counter()
        latency_ms = round((t_end - t_start) * 1000, 4)

        # ── Assemble result dict ──────────────────────────────────────────────
        ts = datetime.datetime.fromtimestamp(feature_dict["_timestamp"])

        result = {
            # ── Flow metadata ─────────────────────────────────────────────────
            "timestamp"  : ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "src_ip"     : feature_dict["_src_ip"],
            "dst_ip"     : feature_dict["_dst_ip"],
            "src_port"   : feature_dict["_src_port"],
            "dst_port"   : feature_dict["_dst_port"],
            "protocol"   : feature_dict["_protocol"],
            "pkt_count"  : feature_dict["_pkt_count"],
            "byte_count" : feature_dict["_byte_count"],
            "duration"   : round(feature_dict["_duration"], 4),
            "pkt_rate"   : round(feature_dict["_pkt_rate"], 4),
            # No ground truth in live mode
            "true_label" : "N/A - Live Traffic",
            # ── XGBoost ──────────────────────────────────────────────────────
            "xgb_pred"   : xgb_pred,
            "xgb_prob"   : round(xgb_prob, 6),
            # ── Autoencoder ──────────────────────────────────────────────────
            "ae_mse"     : round(ae_mse, 6),
            "ae_score"   : round(ae_score, 6),    # normalized [0,1]
            "ae_pred"    : ae_pred,
            # ── GNN / Temporal ────────────────────────────────────────────────
            "gnn_score"      : fusion_result.get("scores", {}).get("gnn"),
            "temporal_score" : fusion_result.get("scores", {}).get("temporal"),
            "gnn_latency_ms" : round(gnn_latency_ms, 3),
            # ── Fusion ───────────────────────────────────────────────────────
            "fusion_score" : fusion_result["fusion_score"],
            "fusion_pred"  : fusion_result["fusion_prediction"],
            "fusion_mode"  : fusion_result["fusion_mode"],
            "fusion_weights" : fusion_result.get("weights", {}),
            "fusion_scores"  : fusion_result.get("scores", {}),
            "fusion_confidence": fusion_result.get("confidence", {}),
            "fusion_consistency": fusion_result.get("consistency", {}),
            "fusion_reliability": fusion_result.get("reliability", {}),
            "available_components": fusion_result.get("available_components", []),
            "missing_components": fusion_result.get("missing_components", []),
            # ── Enriched behavioral features ─────────────────────────────────
            "enriched"       : enriched,
            # ── Performance ──────────────────────────────────────────────────
            "inference_latency_ms": latency_ms,
        }

        return result


# ── Module-level singleton ────────────────────────────────────────────────────
engine = InferenceEngine()
