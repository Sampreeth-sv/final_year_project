"""
fusion_engine.py
================
AI-Powered Network Intrusion Detection System — FUSION ENGINE

Implements the fixed-weight fusion of detection signals from multiple models.

Current available components:
  - XGBoost (active)
  - Autoencoder (active)
  - Bipartite GNN (NOT YET AVAILABLE — Person 1 integration pending)
  - Temporal Correlation (NOT YET AVAILABLE — pending implementation)

Full fusion weights (per architecture spec):
    S_fusion = 0.35 * S_XGB + 0.20 * S_AE + 0.25 * S_GNN + 0.20 * S_Temporal

Partial fusion (XGB + AE only — normalized over available components):
    XGB effective weight = 0.35 / (0.35 + 0.20) = 0.35 / 0.55 ≈ 0.63636
    AE  effective weight = 0.20 / (0.35 + 0.20) = 0.20 / 0.55 ≈ 0.36364

    S_partial = (0.35/0.55) * S_XGB + (0.20/0.55) * S_AE

Score normalization contract:
    All input scores MUST be in [0, 1].
    S_XGB  = XGBoost attack probability
    S_AE   = AE MSE normalized via xgb_ae_mse_max (leakage-free artifact)
    S_GNN  = GNN malicious probability (Person 1 integration, not yet available)
    S_TEMPORAL = Temporal correlation risk score (not yet available)

Usage:
    from fusion_engine import FusionEngine
    fe = FusionEngine(threshold=0.5)
    result = fe.fuse(xgb_score=0.85, ae_score=0.62)
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

# ── Fixed architecture weights (per spec) ─────────────────────────────────────
_WEIGHT_XGB      = 0.35
_WEIGHT_AE       = 0.20
_WEIGHT_GNN      = 0.25
_WEIGHT_TEMPORAL = 0.20
_WEIGHT_TOTAL    = _WEIGHT_XGB + _WEIGHT_AE + _WEIGHT_GNN + _WEIGHT_TEMPORAL

assert abs(_WEIGHT_TOTAL - 1.0) < 1e-9, "Fusion weights must sum to 1.0"

# ── Default threshold config path ─────────────────────────────────────────────
_HERE            = os.path.dirname(os.path.abspath(__file__))
_FUSION_CFG_PATH = os.path.join(_HERE, "..", "models", "fusion_config.json")

_DEFAULT_THRESHOLD = 0.5


def load_fusion_threshold(config_path: str = _FUSION_CFG_PATH) -> float:
    """
    Load the fusion decision threshold from fusion_config.json.
    Falls back to _DEFAULT_THRESHOLD (0.5) if the file does not exist.
    """
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            threshold = float(cfg.get("fusion_threshold", _DEFAULT_THRESHOLD))
            logger.info("Fusion threshold loaded from config: %.4f", threshold)
            return threshold
        except Exception as exc:
            logger.warning(
                "Could not read fusion_config.json (%s) — using default %.2f",
                exc, _DEFAULT_THRESHOLD,
            )
    else:
        logger.info(
            "fusion_config.json not found at %s — using default threshold %.2f",
            config_path, _DEFAULT_THRESHOLD,
        )
    return _DEFAULT_THRESHOLD


class FusionEngine:
    """
    Fixed-weight fusion of XGBoost, Autoencoder, GNN, and Temporal signals.

    Operates in two explicit modes:

    PARTIAL FUSION — GNN/TEMPORAL NOT YET AVAILABLE
        Weights renormalized over the two available components:
            w_XGB_eff = 0.35 / 0.55 ≈ 0.6364
            w_AE_eff  = 0.20 / 0.55 ≈ 0.3636

    FULL FUSION (when all four genuine scores are provided)
        S_fusion = 0.35*S_XGB + 0.20*S_AE + 0.25*S_GNN + 0.20*S_Temporal

    Parameters
    ----------
    threshold : float
        Decision threshold for fusion_prediction (default: 0.5).
        Scores >= threshold → prediction = 1 (attack).
    """

    # Full-fusion weights (architecture spec)
    W_XGB      = _WEIGHT_XGB
    W_AE       = _WEIGHT_AE
    W_GNN      = _WEIGHT_GNN
    W_TEMPORAL = _WEIGHT_TEMPORAL

    # Partial-fusion effective weights (XGB + AE only, renormalized)
    _PARTIAL_DENOM = _WEIGHT_XGB + _WEIGHT_AE  # 0.55
    W_XGB_PARTIAL  = _WEIGHT_XGB / _PARTIAL_DENOM   # 0.35/0.55 ≈ 0.63636
    W_AE_PARTIAL   = _WEIGHT_AE  / _PARTIAL_DENOM   # 0.20/0.55 ≈ 0.36364

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD):
        self.threshold = threshold
        logger.info(
            "FusionEngine initialized. threshold=%.4f  "
            "partial weights: XGB=%.5f AE=%.5f  "
            "full weights: XGB=%.2f AE=%.2f GNN=%.2f Temporal=%.2f",
            self.threshold,
            self.W_XGB_PARTIAL, self.W_AE_PARTIAL,
            self.W_XGB, self.W_AE, self.W_GNN, self.W_TEMPORAL,
        )

    def fuse(
        self,
        xgb_score: float,
        ae_score: float,
        gnn_score: float = None,
        temporal_score: float = None,
    ) -> dict:
        """
        Compute the fusion result from available component scores.

        Parameters
        ----------
        xgb_score : float
            XGBoost attack probability in [0, 1].
        ae_score : float
            AE anomaly score normalized to [0, 1].
        gnn_score : float or None
            GNN malicious probability in [0, 1].
            Pass None until Person 1's GNN is integrated.
            DO NOT pass a fabricated/random value.
        temporal_score : float or None
            Temporal correlation risk score in [0, 1].
            Pass None until the temporal component is implemented.
            DO NOT pass a fabricated/random value.

        Returns
        -------
        dict with keys:
            xgb_score, ae_score, gnn_score, temporal_score,
            available_components,
            fusion_score, fusion_prediction, fusion_mode
        """
        # ── Input validation ──────────────────────────────────────────────────
        for name, val in [("xgb_score", xgb_score), ("ae_score", ae_score)]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"FusionEngine.fuse(): {name}={val!r} is not in [0, 1]. "
                    "All scores must be normalized before fusion."
                )

        gnn_available      = gnn_score      is not None
        temporal_available = temporal_score is not None

        if gnn_available and not (0.0 <= gnn_score <= 1.0):
            raise ValueError(
                f"FusionEngine.fuse(): gnn_score={gnn_score!r} is not in [0, 1]."
            )
        if temporal_available and not (0.0 <= temporal_score <= 1.0):
            raise ValueError(
                f"FusionEngine.fuse(): temporal_score={temporal_score!r} is not in [0, 1]."
            )

        available_components = {
            "xgb":      True,
            "ae":       True,
            "gnn":      gnn_available,
            "temporal": temporal_available,
        }

        # ── Fusion calculation ────────────────────────────────────────────────
        if gnn_available and temporal_available:
            # Full fusion: all four components present
            fusion_score = (
                self.W_XGB      * xgb_score
                + self.W_AE       * ae_score
                + self.W_GNN      * gnn_score
                + self.W_TEMPORAL * temporal_score
            )
            fusion_mode = "full"

        else:
            # Partial fusion: ONLY XGB + AE (renormalized weights)
            # Clearly labelled: PARTIAL FUSION — GNN/TEMPORAL NOT YET AVAILABLE
            fusion_score = (
                self.W_XGB_PARTIAL * xgb_score
                + self.W_AE_PARTIAL  * ae_score
            )
            fusion_mode = "partial"

        fusion_score = round(float(fusion_score), 6)
        fusion_prediction = int(fusion_score >= self.threshold)

        return {
            # Individual component scores (None if not yet available)
            "xgb_score"      : round(float(xgb_score), 6),
            "ae_score"       : round(float(ae_score),  6),
            "gnn_score"      : round(float(gnn_score), 6) if gnn_available else None,
            "temporal_score" : round(float(temporal_score), 6) if temporal_available else None,
            # Component availability
            "available_components": available_components,
            # Fusion result
            "fusion_score"      : fusion_score,
            "fusion_prediction" : fusion_prediction,
            "fusion_mode"       : fusion_mode,
        }

    @staticmethod
    def compute_partial_fusion(xgb_score: float, ae_score: float) -> float:
        """
        Convenience method: compute partial fusion score directly.

        Returns the scalar partial fusion score (not a full result dict).
        Useful for testing the formula in isolation.
        """
        return (
            FusionEngine.W_XGB_PARTIAL * xgb_score
            + FusionEngine.W_AE_PARTIAL  * ae_score
        )

    @staticmethod
    def compute_full_fusion(
        xgb_score: float,
        ae_score: float,
        gnn_score: float,
        temporal_score: float,
    ) -> float:
        """
        Convenience method: compute full fusion score directly.

        Formula: 0.35*xgb + 0.20*ae + 0.25*gnn + 0.20*temporal

        Returns the scalar full fusion score.
        Useful for testing the formula in isolation.
        """
        return (
            FusionEngine.W_XGB      * xgb_score
            + FusionEngine.W_AE       * ae_score
            + FusionEngine.W_GNN      * gnn_score
            + FusionEngine.W_TEMPORAL * temporal_score
        )
