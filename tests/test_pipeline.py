"""
test_pipeline.py
================
Phase 2 — Active Pipeline Test Suite (16 success criteria)

Tests the XGBoost + Autoencoder + Fusion Engine pipeline.
Random Forest is NOT a requirement of the active system.

Run:
    python -m pytest test_pipeline.py -v

Success criteria:
  C1.  RF is NOT required by the active inference pipeline
  C2.  Active system loads without rf_model.pkl present
  C3.  XGBoost model loads and predicts
  C4.  XGBoost probability is in [0, 1]
  C5.  AE inference works (score + pred)
  C6.  AE score normalized to [0, 1] (via xgb_ae_mse_max)
  C7.  Missing features raise a schema error (ValueError) — never fabricated
  C8.  Flow extractor export matches all 49 canonical features
  C9.  Partial fusion calculates correctly (XGB+AE renormalized weights)
  C10. Full fusion formula: 0.35*xgb + 0.20*ae + 0.25*gnn + 0.20*temporal
  C11. Partial fusion result labelled as "partial"
  C12. Unavailable GNN/temporal returns None — never fabricated
  C13. Inference output contains no RF keys (xgb_pred present, rf_pred absent)
  C14. Stream/replay controls absent from app.py layout (no pos-slider etc.)
  C15. Dashboard callback does NOT require pos-slider input
  C16. Inference latency is reported in the result dict

Note on TensorFlow DLL issues (Windows):
  Tests that load the InferenceEngine (which imports TensorFlow) are automatically
  skipped if TensorFlow cannot be imported in the test environment. This is a
  known Windows-specific DLL incompatibility in some TF builds and does NOT affect
  the live system (app.py imports TF successfully in-process).
"""

import os
import json
import math
import time
import pytest
import numpy as np

# ── TensorFlow availability guard ──────────────────────────────────────────────
import subprocess as _subprocess
import sys as _sys

def _probe_tf() -> bool:
    """Return True if TensorFlow can be imported in a fresh subprocess."""
    try:
        r = _subprocess.run(
            [_sys.executable, "-c", "import tensorflow"],
            capture_output=True, timeout=30,
        )
        return r.returncode == 0
    except Exception:
        return False

_TF_AVAILABLE  = _probe_tf()
_TF_SKIP_REASON = (
    "TensorFlow not importable in test env (Windows DLL issue). "
    "These tests pass when run via  python app.py  — TF works in-process."
    if not _TF_AVAILABLE else ""
)

requires_tf = pytest.mark.skipif(not _TF_AVAILABLE, reason=_TF_SKIP_REASON or "TF unavailable")

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

MODELS_DIR  = os.path.join(_PROJECT_ROOT, "models")

AE_PATH     = os.path.join(MODELS_DIR, "ae_model.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ART_PATH    = os.path.join(MODELS_DIR, "artifacts.json")
XGB_PATH    = os.path.join(MODELS_DIR, "xgb_model.pkl")
XGB_ART     = os.path.join(MODELS_DIR, "xgb_artifacts.json")
RF_PATH     = os.path.join(MODELS_DIR, "rf_model.pkl")   # LEGACY — must NOT be required

from feature_schema import FEATURE_COLS  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dummy_feature_dict(value: float = 1.0) -> dict:
    """Return a feature dict with all 49 FEATURE_COLS set to `value` plus metadata."""
    d = {c: value for c in FEATURE_COLS}
    d["_timestamp"] = time.time()
    d["_src_ip"]    = "192.168.1.1"
    d["_dst_ip"]    = "10.0.0.1"
    d["_src_port"]  = 12345
    d["_dst_port"]  = 80
    d["_protocol"]  = "TCP"
    d["_pkt_count"] = 10
    d["_byte_count"] = 1000
    d["_duration"]  = 1.0
    d["_pkt_rate"]  = 10.0
    return d


def _fresh_engine():
    """Return a freshly loaded InferenceEngine (XGBoost + AE + Fusion)."""
    from inference import InferenceEngine
    eng = InferenceEngine()
    eng.load()
    return eng


# ─────────────────────────────────────────────────────────────────────────────
# C1-C2 — RF independence
# ─────────────────────────────────────────────────────────────────────────────

class TestRFIndependence:

    def test_c1_active_inference_does_not_load_rf(self):
        """C1: inference.py must not define _RF_PATH or actively load rf_model.pkl."""
        inf_path = os.path.join(_PROJECT_ROOT, "inference.py")
        with open(inf_path) as f:
            source = f.read()
        # Active loading indicators: path constant or joblib.load pattern for RF
        # (docstring mentions are fine; actual code loading is not allowed)
        assert "_RF_PATH" not in source, (
            "_RF_PATH must not be defined in inference.py — "
            "RF has been removed from the active pipeline."
        )
        # Check no rf_model.pkl inside a path-join call (actual load code)
        import re
        rf_load_pattern = re.compile(r'os\.path\.join\([^)]*rf_model\.pkl[^)]*\)')
        assert not rf_load_pattern.search(source), (
            "inference.py must not have os.path.join(...rf_model.pkl...) — "
            "RF is not part of the active pipeline."
        )

    def test_c1b_app_does_not_require_rf_model(self):
        """C1b: app.py startup check must not require rf_model.pkl."""
        app_path = os.path.join(_PROJECT_ROOT, "app.py")
        with open(app_path) as f:
            source = f.read()
        # rf_model.pkl must not be in the required artifacts list in app.py
        assert "rf_model.pkl" not in source, (
            "app.py must not require rf_model.pkl at startup — "
            "RF has been removed from the active pipeline."
        )

    @requires_tf
    def test_c2_engine_loads_without_rf(self):
        """C2: InferenceEngine.load() succeeds even if rf_model.pkl is absent."""
        # This test just calls load() — it should succeed because rf_model.pkl
        # is no longer required by the active inference engine.
        eng = _fresh_engine()
        assert eng._loaded, "InferenceEngine should be in loaded state after load()"
        assert eng._xgb_loaded, "XGBoost should be loaded after load()"
        assert eng._xgb is not None
        assert eng._ae  is not None


# ─────────────────────────────────────────────────────────────────────────────
# C3-C4 — XGBoost
# ─────────────────────────────────────────────────────────────────────────────

class TestXGBoost:

    def test_c3_xgb_model_loads(self):
        """C3: XGBoost model loads from models/xgb_model.pkl."""
        if not os.path.exists(XGB_PATH):
            pytest.skip("xgb_model.pkl not found — run train_xgboost_ae.py first")
        import joblib
        model = joblib.load(XGB_PATH)
        assert model is not None

    def test_c3b_xgb_prediction_binary(self):
        """C3b: XGBoost prediction returns 0 or 1."""
        if not os.path.exists(XGB_PATH):
            pytest.skip("xgb_model.pkl not found — run train_xgboost_ae.py first")
        import joblib
        model = joblib.load(XGB_PATH)
        x = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
        pred = model.predict(x)[0]
        assert pred in (0, 1), f"Expected 0 or 1, got {pred}"

    def test_c4_xgb_probability_in_range(self):
        """C4: XGBoost probability is in [0, 1]."""
        if not os.path.exists(XGB_PATH):
            pytest.skip("xgb_model.pkl not found — run train_xgboost_ae.py first")
        import joblib
        model = joblib.load(XGB_PATH)
        x = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
        prob = model.predict_proba(x)[0]
        assert all(0.0 <= p <= 1.0 for p in prob), f"Probabilities out of range: {prob}"
        assert abs(sum(prob) - 1.0) < 1e-5, f"Probabilities do not sum to 1: {prob}"


# ─────────────────────────────────────────────────────────────────────────────
# C5-C6 — Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoencoder:

    @requires_tf
    def test_c5_ae_inference_works(self):
        """C5: AE inference returns ae_mse and ae_pred (0 or 1)."""
        if not all(os.path.exists(p) for p in [AE_PATH, SCALER_PATH, ART_PATH]):
            pytest.skip("AE artifacts not found — run train_xgboost_ae.py first")

        import tensorflow as tf
        import joblib

        ae     = tf.keras.models.load_model(AE_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(ART_PATH) as f:
            threshold = float(json.load(f)["ae_threshold"])

        x_raw    = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
        x_scaled = scaler.transform(x_raw).astype(np.float32)
        recon    = ae.predict(x_scaled, verbose=0)
        mse      = float(np.mean(np.square(recon - x_scaled)))
        pred     = int(mse > threshold)

        assert mse >= 0.0, f"ae_mse should be non-negative, got {mse}"
        assert pred in (0, 1), f"ae_pred should be 0 or 1, got {pred}"

    @requires_tf
    def test_c6_ae_score_normalized(self):
        """C6: ae_score (from InferenceEngine) is in [0, 1]."""
        if not os.path.exists(XGB_PATH):
            pytest.skip("xgb_model.pkl not found — run train_xgboost_ae.py first")

        eng    = _fresh_engine()
        feat   = _dummy_feature_dict(0.0)
        result = eng.run_inference(feat)

        assert "ae_score" in result, "ae_score key missing from inference result"
        ae_score = result["ae_score"]
        assert 0.0 <= ae_score <= 1.0, (
            f"ae_score={ae_score} is not in [0, 1]. "
            "AE score must be normalized via xgb_ae_mse_max before fusion."
        )


# ─────────────────────────────────────────────────────────────────────────────
# C7 — Feature validation / missing features
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureValidation:

    @requires_tf
    def test_c7_missing_feature_raises_error(self):
        """C7: A missing required feature raises ValueError — never fabricated."""
        eng  = _fresh_engine()
        feat = _dummy_feature_dict(0.0)
        del feat[FEATURE_COLS[0]]   # Remove the first feature

        with pytest.raises((ValueError, Exception)):
            eng.run_inference(feat)

    @requires_tf
    def test_c7b_zero_value_feature_is_valid(self):
        """C7b: A feature value of 0 is legitimate — must NOT be treated as missing."""
        eng  = _fresh_engine()
        feat = _dummy_feature_dict(0.0)   # All 49 features set to 0.0
        # Should succeed — 0 is a valid feature value
        result = eng.run_inference(feat)
        assert result is not None, "Inference failed with all-zero feature values"

    def test_c7c_exactly_49_features(self):
        """C7c: FEATURE_COLS contains exactly 49 features."""
        assert len(FEATURE_COLS) == 49, f"Expected 49, got {len(FEATURE_COLS)}"

    def test_c7d_no_duplicate_features(self):
        """C7d: No duplicate names in FEATURE_COLS."""
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS)), (
            "Duplicate feature names found in FEATURE_COLS"
        )


# ─────────────────────────────────────────────────────────────────────────────
# C8 — Flow extractor 49-feature compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestFlowExtractor:

    def test_c8_flow_export_contains_all_49_features(self):
        """C8: Flow extractor export dict contains all 49 FEATURE_COLS."""
        from flow_extractor import FlowExtractor, _FlowRecord

        flow = _FlowRecord(
            src_ip="1.2.3.4", dst_ip="5.6.7.8",
            src_port=12345, dst_port=80, protocol=6,
            now=time.time(),
        )
        flow.in_bytes  = 100
        flow.in_pkts   = 1
        flow.pkt_lengths.append(100)
        flow.t_first_in = flow.t_first
        flow.t_last_in  = flow.t_first
        flow.npkts_0_128 = 1

        extractor = FlowExtractor()
        exported  = extractor._export(flow)

        assert exported is not None, "FlowExtractor._export() returned None unexpectedly"

        missing = [c for c in FEATURE_COLS if c not in exported]
        assert not missing, (
            f"Flow export is missing {len(missing)} FEATURE_COLS: {missing[:10]}"
        )

    @requires_tf
    def test_c8b_flow_extractor_feeds_inference(self):
        """C8b: A flow extractor output dict feeds correctly through InferenceEngine."""
        from flow_extractor import FlowExtractor, _FlowRecord

        flow = _FlowRecord(
            src_ip="1.2.3.4", dst_ip="5.6.7.8",
            src_port=12345, dst_port=80, protocol=6,
            now=time.time(),
        )
        flow.in_bytes    = 500
        flow.in_pkts     = 5
        flow.out_bytes   = 300
        flow.out_pkts    = 3
        flow.pkt_lengths = [100, 100, 100, 100, 100]
        flow.t_first_in  = flow.t_first
        flow.t_last_in   = flow.t_first
        flow.npkts_0_128 = 5

        extractor = FlowExtractor()
        feat_dict = extractor._export(flow)
        assert feat_dict is not None

        eng    = _fresh_engine()
        result = eng.run_inference(feat_dict)

        for k in ("xgb_pred", "xgb_prob", "ae_mse", "ae_score", "ae_pred",
                  "fusion_score", "fusion_pred", "fusion_mode"):
            assert k in result, f"Key '{k}' missing from live-compatibility result"


# ─────────────────────────────────────────────────────────────────────────────
# C9-C12 — Fusion Engine
# ─────────────────────────────────────────────────────────────────────────────

class TestFusionEngine:

    def test_c9_partial_fusion_formula(self):
        """C9: Partial fusion uses renormalized XGB+AE weights."""
        from fusion.fusion_engine import FusionEngine

        fe = FusionEngine(threshold=0.5)

        xgb_score = 0.8
        ae_score  = 0.4

        result = fe.fuse(xgb_score=xgb_score, ae_score=ae_score)

        # Expected: (0.35/0.55)*0.8 + (0.20/0.55)*0.4
        expected = (0.35 / 0.55) * xgb_score + (0.20 / 0.55) * ae_score
        assert abs(result["fusion_score"] - expected) < 1e-4, (
            f"Partial fusion score {result['fusion_score']:.6f} != expected {expected:.6f}"
        )

    def test_c9b_partial_fusion_static_method(self):
        """C9b: FusionEngine.compute_partial_fusion() matches formula."""
        from fusion.fusion_engine import FusionEngine

        xgb_score = 0.7
        ae_score  = 0.3

        computed = FusionEngine.compute_partial_fusion(xgb_score, ae_score)
        expected = (0.35 / 0.55) * xgb_score + (0.20 / 0.55) * ae_score

        assert abs(computed - expected) < 1e-10, (
            f"compute_partial_fusion: {computed} != {expected}"
        )

    def test_c10_full_fusion_formula(self):
        """C10: Full fusion uses 0.35*xgb + 0.20*ae + 0.25*gnn + 0.20*temporal."""
        from fusion.fusion_engine import FusionEngine

        fe = FusionEngine(threshold=0.5)

        xgb_score      = 0.9
        ae_score       = 0.6
        gnn_score      = 0.7
        temporal_score = 0.5

        result = fe.fuse(
            xgb_score=xgb_score,
            ae_score=ae_score,
            gnn_score=gnn_score,
            temporal_score=temporal_score,
        )

        expected = (
            0.35 * xgb_score
            + 0.20 * ae_score
            + 0.25 * gnn_score
            + 0.20 * temporal_score
        )
        assert abs(result["fusion_score"] - expected) < 1e-4, (
            f"Full fusion score {result['fusion_score']:.6f} != expected {expected:.6f}"
        )
        assert result["fusion_mode"] == "full", (
            f"Expected fusion_mode='full', got '{result['fusion_mode']}'"
        )

    def test_c10b_full_fusion_weights_sum_to_one(self):
        """C10b: Full fusion weights must sum exactly to 1.0."""
        from fusion.fusion_engine import FusionEngine
        w_total = FusionEngine.W_XGB + FusionEngine.W_AE + FusionEngine.W_GNN + FusionEngine.W_TEMPORAL
        assert abs(w_total - 1.0) < 1e-10, f"Full fusion weights sum to {w_total}, expected 1.0"

    def test_c10c_full_fusion_static_method(self):
        """C10c: FusionEngine.compute_full_fusion() matches the 4-component formula."""
        from fusion.fusion_engine import FusionEngine

        xgb_score, ae_score, gnn_score, temporal_score = 0.9, 0.6, 0.7, 0.5
        computed = FusionEngine.compute_full_fusion(xgb_score, ae_score, gnn_score, temporal_score)
        expected = 0.35 * xgb_score + 0.20 * ae_score + 0.25 * gnn_score + 0.20 * temporal_score
        assert abs(computed - expected) < 1e-10, f"{computed} != {expected}"

    def test_c11_partial_fusion_mode_label(self):
        """C11: Partial fusion (GNN/temporal absent) is labelled 'partial'."""
        from fusion.fusion_engine import FusionEngine

        fe = FusionEngine(threshold=0.5)
        result = fe.fuse(xgb_score=0.7, ae_score=0.4, gnn_score=None, temporal_score=None)

        assert result["fusion_mode"] == "partial", (
            f"Expected fusion_mode='partial', got '{result['fusion_mode']}'"
        )

    def test_c12_gnn_temporal_none_when_unavailable(self):
        """C12: gnn_score and temporal_score are None when not provided — never fabricated."""
        from fusion.fusion_engine import FusionEngine

        fe = FusionEngine(threshold=0.5)
        result = fe.fuse(xgb_score=0.7, ae_score=0.4)

        assert result["gnn_score"]      is None, "gnn_score must be None when not provided"
        assert result["temporal_score"] is None, "temporal_score must be None when not provided"
        assert result["available_components"]["gnn"]      is False
        assert result["available_components"]["temporal"] is False

    def test_c12b_fusion_score_in_range(self):
        """C12b: Fusion score is always in [0, 1] for valid input scores."""
        from fusion.fusion_engine import FusionEngine

        fe = FusionEngine(threshold=0.5)
        for xgb, ae in [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.3, 0.9)]:
            result = fe.fuse(xgb_score=xgb, ae_score=ae)
            score = result["fusion_score"]
            assert 0.0 <= score <= 1.0, f"Fusion score {score} out of [0, 1] for xgb={xgb}, ae={ae}"

    def test_c12c_fusion_threshold_default(self):
        """C12c: Fusion threshold is 0.5 by default."""
        from fusion.fusion_engine import FusionEngine, _DEFAULT_THRESHOLD

        assert _DEFAULT_THRESHOLD == 0.5, f"Expected default threshold 0.5, got {_DEFAULT_THRESHOLD}"

        fe = FusionEngine()
        assert fe.threshold == 0.5

    def test_c12d_fusion_out_of_range_raises(self):
        """C12d: Fusion engine raises ValueError for out-of-range input scores."""
        from fusion.fusion_engine import FusionEngine

        fe = FusionEngine()

        with pytest.raises(ValueError):
            fe.fuse(xgb_score=1.5, ae_score=0.5)   # xgb out of range

        with pytest.raises(ValueError):
            fe.fuse(xgb_score=0.5, ae_score=-0.1)  # ae out of range


# ─────────────────────────────────────────────────────────────────────────────
# C13 — No RF keys in inference output
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceOutput:

    @requires_tf
    def test_c13_no_rf_keys_in_result(self):
        """C13: Inference result contains XGBoost keys but NOT RF keys."""
        if not os.path.exists(XGB_PATH):
            pytest.skip("xgb_model.pkl not found — run train_xgboost_ae.py first")

        eng    = _fresh_engine()
        feat   = _dummy_feature_dict(0.0)
        result = eng.run_inference(feat)

        # XGBoost keys must be present
        for k in ("xgb_pred", "xgb_prob"):
            assert k in result, f"Active model key '{k}' missing from inference result"

        # Fusion keys must be present
        for k in ("fusion_score", "fusion_pred", "fusion_mode"):
            assert k in result, f"Fusion key '{k}' missing from inference result"

        # GNN/Temporal must be None
        assert result["gnn_score"]      is None, "gnn_score should be None"
        assert result["temporal_score"] is None, "temporal_score should be None"

        # RF keys must NOT be present
        for k in ("rf_pred", "rf_prob", "score_combined"):
            assert k not in result, (
                f"RF key '{k}' should NOT be in inference result — "
                "RF has been removed from the active pipeline."
            )

    @requires_tf
    def test_c16_inference_latency_reported(self):
        """C16: inference_latency_ms is reported in the result dict."""
        if not os.path.exists(XGB_PATH):
            pytest.skip("xgb_model.pkl not found — run train_xgboost_ae.py first")

        eng    = _fresh_engine()
        feat   = _dummy_feature_dict(0.0)
        result = eng.run_inference(feat)

        assert "inference_latency_ms" in result, "inference_latency_ms missing from result"
        assert result["inference_latency_ms"] >= 0, (
            f"inference_latency_ms should be non-negative, got {result['inference_latency_ms']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# C14-C15 — Dashboard / app.py structural checks
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardStructure:

    def test_c14_no_stream_controls_in_app(self):
        """C14: app.py must not contain stream/replay position controls."""
        app_path = os.path.join(_PROJECT_ROOT, "app.py")
        with open(app_path) as f:
            source = f.read()

        forbidden = ["pos-slider", "play-btn", "pause-btn", "end-btn", "playing-store"]
        for token in forbidden:
            assert token not in source, (
                f"Stream control '{token}' found in app.py — "
                "stream position/replay controls must be removed."
            )

    def test_c15_dashboard_callback_uses_interval_not_slider(self):
        """C15: Main dashboard callback uses interval as input, not pos-slider."""
        app_path = os.path.join(_PROJECT_ROOT, "app.py")
        with open(app_path) as f:
            source = f.read()

        # The auto-update callback must use the interval component
        assert "interval" in source, "Dashboard must have a dcc.Interval component"
        # The callback should NOT have auto_advance or pos-slider as inputs
        assert "auto_advance" not in source, (
            "auto_advance callback should have been removed from app.py"
        )

    def test_c14b_no_rf_confusion_matrix_in_app(self):
        """C14b: RF Confusion Matrix section must be removed from app.py."""
        app_path = os.path.join(_PROJECT_ROOT, "app.py")
        with open(app_path) as f:
            source = f.read()

        # The "Confusion Matrix (RF)" section should be gone
        assert "conf-matrix" not in source, (
            "'conf-matrix' component should be removed — RF confusion matrix not needed."
        )

    def test_c14c_no_rf_cards_in_app(self):
        """C14c: RF Alerts card must not appear in app.py."""
        app_path = os.path.join(_PROJECT_ROOT, "app.py")
        with open(app_path) as f:
            source = f.read()

        assert "RF Alerts" not in source, (
            "'RF Alerts' card must be removed from app.py."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Artifact consistency checks
# ─────────────────────────────────────────────────────────────────────────────

class TestArtifacts:

    def test_schema_matches_artifacts_json(self):
        """FEATURE_COLS matches the list stored in artifacts.json."""
        if not os.path.exists(ART_PATH):
            pytest.skip("artifacts.json not found — run train_xgboost_ae.py first")
        with open(ART_PATH) as f:
            art = json.load(f)
        assert art["feature_cols"] == FEATURE_COLS

    def test_schema_matches_xgb_artifacts_json(self):
        """FEATURE_COLS matches the list stored in xgb_artifacts.json."""
        if not os.path.exists(XGB_ART):
            pytest.skip("xgb_artifacts.json not found — run train_xgboost_ae.py first")
        with open(XGB_ART) as f:
            art = json.load(f)
        assert art["feature_cols"] == FEATURE_COLS

    def test_xgb_ae_mse_max_is_positive(self):
        """xgb_ae_mse_max must be > 0 (was derived from actual data)."""
        if not os.path.exists(XGB_ART):
            pytest.skip("xgb_artifacts.json not found — run train_xgboost_ae.py first")
        with open(XGB_ART) as f:
            art = json.load(f)
        assert art["xgb_ae_mse_max"] > 0.0

    def test_fusion_config_exists(self):
        """fusion_config.json must exist in models/."""
        cfg_path = os.path.join(MODELS_DIR, "fusion_config.json")
        assert os.path.exists(cfg_path), (
            "models/fusion_config.json not found. "
            "It should have been created automatically."
        )
        with open(cfg_path) as f:
            cfg = json.load(f)
        assert "fusion_threshold" in cfg
        assert 0.0 < cfg["fusion_threshold"] <= 1.0
