"""
fusion/fusion_engine.py
=======================
Dynamic Multi-Model Fusion Engine

Implements entropy-based confidence and consistency-weighted dynamic fusion
of detection signals from XGBoost, Autoencoder, GNN, and Temporal models.

Core equations:
- Confidence: C = 1 - H_norm(p) where H_norm(p) = H(p)/log(2) for probabilities
- Consistency: Cons_i = 1 - normalized(|S_i - mean_score|)
- Reliability: R_i = C_i × Cons_i
- Dynamic weight: w_i = R_i / Σ R_j
- Full fusion: S_fusion = Σ w_i × S_i

Author: Claude Code | Hybrid Architecture Plan
"""

import math
import json
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Default configuration
_DEFAULT_THRESHOLD = 0.5
# Temporal normalization factor fallback (only used if fusion_config.json
# does not contain a calibrated temporal_max_norm). The actual value used
# at runtime is loaded from models/fusion_config.json, which is calibrated
# from validation data by calibrate_temporal_norm.py.
_DEFAULT_TEMPORAL_MAX_NORM = 100.0

# Config path
_HERE = os.path.dirname(os.path.abspath(__file__))
_FUSION_CFG_PATH = os.path.join(_HERE, "..", "models", "fusion_config.json")


def load_fusion_threshold(config_path: str = _FUSION_CFG_PATH) -> float:
    """
    Load the fusion decision threshold from fusion_config.json.
    Falls back to _DEFAULT_THRESHOLD (0.5) if the file does not exist.

    Parameters
    ----------
    config_path : str
        Path to the fusion configuration JSON file

    Returns
    -------
    float
        The fusion threshold value
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


def load_temporal_max_norm(config_path: str = _FUSION_CFG_PATH) -> float:
    """
    Load the calibrated temporal_max_norm from fusion_config.json.

    This value is the 95th percentile of GRU hidden-state L2 norms computed
    from the validation set by calibrate_temporal_norm.py. It is used by
    normalize_temporal_score() to map raw L2 norms into [0, 1].

    Falls back to _DEFAULT_TEMPORAL_MAX_NORM (100.0) only if the config file
    does not exist or the key is missing.

    Parameters
    ----------
    config_path : str
        Path to the fusion configuration JSON file

    Returns
    -------
    float
        The calibrated temporal_max_norm
    """
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            max_norm = float(cfg.get("temporal_max_norm", _DEFAULT_TEMPORAL_MAX_NORM))
            logger.info(
                "Temporal max_norm loaded from config: %.6f (calibrated from validation data)",
                max_norm,
            )
            return max_norm
        except Exception as exc:
            logger.warning(
                "Could not read temporal_max_norm from fusion_config.json (%s) — "
                "using default %.2f",
                exc, _DEFAULT_TEMPORAL_MAX_NORM,
            )
    else:
        logger.info(
            "fusion_config.json not found at %s — using default temporal max_norm %.2f. "
            "Run calibrate_temporal_norm.py to derive a data-driven value.",
            config_path, _DEFAULT_TEMPORAL_MAX_NORM,
        )
    return _DEFAULT_TEMPORAL_MAX_NORM


# =============================================================================
# SCORE NORMALIZATION
# =============================================================================

def normalize_temporal_score(raw_norm: float, max_norm: Optional[float] = None) -> float:
    """
    Normalize raw GRU hidden state L2 norm to [0, 1].

    The temporal signal is the L2 norm of GRU hidden states.
    Without normalization, it can be arbitrarily large.

    Uses a logistic (sigmoid) normalization:
        S(x) = 1 / (1 + exp(-k * (x - x0)))

    where x0 = max_norm/2 (midpoint) and k controls steepness.
    This maps:
        - norm ≈ 0 → score ≈ 0.007 (essentially 0)
        - norm ≈ max_norm/2 → score ≈ 0.5 (neutral)
        - norm ≈ max_norm → score ≈ 0.993 (essentially 1)

    CALIBRATION: max_norm is loaded from models/fusion_config.json, which
    is the 95th percentile of temporal norms computed from the validation
    set by calibrate_temporal_norm.py. The fallback default is 100.0, but
    this only applies if the config file is missing or has no
    temporal_max_norm key.

    Parameters
    ----------
    raw_norm : float
        Raw L2 norm of GRU hidden state
    max_norm : float, optional
        Expected maximum norm for scaling. If None, loaded from
        models/fusion_config.json. Default None (load from config).

    Returns
    -------
    float
        Normalized temporal score in [0, 1]
    """
    if max_norm is None:
        max_norm = load_temporal_max_norm()

    norm = max(0.0, raw_norm)
    # Sigmoid centered at max_norm/2 with steepness k = 4/max_norm
    k = 4.0 / max(max_norm, 1e-6)
    x0 = max_norm / 2.0
    normalized = 1.0 / (1.0 + math.exp(-k * (norm - x0)))
    return float(normalized)


# =============================================================================
# ENTROPY-BASED CONFIDENCE
# =============================================================================

def calculate_entropy(p: float) -> float:
    """
    Calculate binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)

    Returns 0 for p=0 or p=1 (certain), 1.0 for p=0.5 (maximum uncertainty).
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    # Clamp to avoid log(0)
    p = max(1e-10, min(1 - 1e-10, p))
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def calculate_normalized_entropy(p: float) -> float:
    """
    Normalize entropy to [0, 1] by dividing by log2(2) = 1.

    For binary entropy, this is just H(p) since log2(2) = 1.

    Returns:
        Normalized entropy: 0 (certain) to 1 (maximum uncertainty)
    """
    return calculate_entropy(p)


def calculate_confidence(p: float) -> float:
    """
    Calculate confidence from probability using entropy.

    Confidence = 1 - H_norm(p)

    - p ≈ 0 or p ≈ 1 → high confidence (low entropy)
    - p ≈ 0.5 → low confidence (high entropy)

    Returns:
        Confidence in [0, 1]
    """
    return 1.0 - calculate_normalized_entropy(p)


# =============================================================================
# CONSISTENCY CALCULATION
# =============================================================================

def calculate_consistency(score: float, other_scores: list) -> float:
    """
    Calculate consistency of a score relative to other available scores.

    Consistency = 1 - normalized_distance
    where normalized_distance = |score - mean| / (max - min + ε)

    A detector is consistent when its score is close to the mean of
    all available scores (agreement with other evidence).

    Parameters
    ----------
    score : float
        The score to evaluate for consistency
    other_scores : list of float
        Other available scores for comparison

    Returns
    -------
    float
        Consistency in [0, 1]
    """
    if not other_scores:
        return 1.0  # No other signals to be consistent with

    all_scores = [score] + other_scores
    min_s = min(all_scores)
    max_s = max(all_scores)
    range_s = max_s - min_s

    if range_s < 1e-10:  # All scores identical
        return 1.0

    mean_score = sum(all_scores) / len(all_scores)
    distance = abs(score - mean_score)
    normalized_distance = distance / range_s

    consistency = 1.0 - normalized_distance
    return max(0.0, consistency)


# =============================================================================
# RELIABILITY CALCULATION
# =============================================================================

def calculate_reliability(confidence: float, consistency: float) -> float:
    """
    Calculate reliability as the product of confidence and consistency.

    R = C × Cons

    A detector is reliable when it has both:
    - High confidence in its own assessment
    - Agreement with other available detectors
    """
    return confidence * consistency


# =============================================================================
# DYNAMIC WEIGHT CALCULATION
# =============================================================================

def calculate_dynamic_weights(
    xgb_prob: float,
    ae_score: float,
    gnn_score: Optional[float] = None,
    temporal_score: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate dynamic fusion weights based on confidence and consistency.

    For each detector:
    1. Calculate confidence (entropy-based for probabilities, proxy for non-probabilities)
    2. Calculate consistency relative to other detectors
    3. Compute reliability = confidence × consistency
    4. Normalize reliabilities to get weights: w_i = R_i / Σ R_j

    Returns
    -------
    dict with keys: xgb, ae, gnn, temporal (None if not available)
    """
    # --- XGBoost: direct probability, entropy-based confidence ---
    c_xgb = calculate_confidence(xgb_prob)

    # --- Autoencoder: anomaly score [0,1], use distance from 0.5 as confidence proxy ---
    # An AE score near 0 or 1 is more confident (clearly normal or anomalous)
    # Score near 0.5 is ambiguous
    # Formula: 2 * |x - 0.5| maps [0,1] to [0,1], min at 0.5, max at 0/1
    ae_certainty = 2 * abs(ae_score - 0.5)
    c_ae = max(0.0, min(1.0, ae_certainty))

    # Available scores for consistency calculation
    available_scores = [xgb_prob, ae_score]
    available_names = ['xgb', 'ae']

    # --- GNN: if available, it's a probability from sigmoid ---
    if gnn_score is not None:
        c_gnn = calculate_confidence(gnn_score)
        available_scores.append(gnn_score)
        available_names.append('gnn')
    else:
        c_gnn = 0.0

    # --- Temporal: if available, normalize first, then use confidence proxy ---
    if temporal_score is not None:
        # temporal_score is raw L2 norm - should be normalized before this function
        # If passed as raw norm, normalize it
        if temporal_score > 1.0:
            t_normalized = normalize_temporal_score(temporal_score)
        else:
            t_normalized = temporal_score

        # Confidence: 2 * |t_normalized - 0.5| maps [0,1] to [0,1],
        # min at 0.5 (uncertain), max at 0/1 (clear)
        c_temp = max(0.0, min(1.0, 2 * abs(t_normalized - 0.5)))
        available_scores.append(t_normalized)
        available_names.append('temporal')
    else:
        t_normalized = None
        c_temp = 0.0

    # --- Calculate consistency ---
    # Each detector's consistency relative to others
    cons_xgb = calculate_consistency(xgb_prob, available_scores[1:])
    cons_ae = calculate_consistency(ae_score, [xgb_prob])

    if gnn_score is not None:
        cons_gnn = calculate_consistency(gnn_score, available_scores[:2])
    else:
        cons_gnn = 0.0

    if temporal_score is not None:
        cons_temp = calculate_consistency(t_normalized, [xgb_prob, ae_score])
    else:
        cons_temp = 0.0

    # --- Calculate reliabilities ---
    r_xgb = calculate_reliability(c_xgb, cons_xgb)
    r_ae = calculate_reliability(c_ae, cons_ae)
    r_gnn = calculate_reliability(c_gnn, cons_gnn) if gnn_score is not None else 0.0
    r_temp = calculate_reliability(c_temp, cons_temp) if temporal_score is not None else 0.0

    # --- Normalize to weights ---
    total_r = r_xgb + r_ae
    if gnn_score is not None:
        total_r += r_gnn
    if temporal_score is not None:
        total_r += r_temp

    if total_r < 1e-10:
        # Fallback: equal weights if all reliabilities are zero
        # Must normalize so weights sum to 1.0
        n_available = 2 + (1 if gnn_score is not None else 0) + (1 if temporal_score is not None else 0)
        w_xgb  = 1.0 / n_available
        w_ae   = 1.0 / n_available
        w_gnn  = 1.0 / n_available if gnn_score is not None else 0.0
        w_temp = 1.0 / n_available if temporal_score is not None else 0.0
    else:
        w_xgb = r_xgb / total_r
        w_ae = r_ae / total_r
        w_gnn = r_gnn / total_r if gnn_score is not None else 0.0
        w_temp = r_temp / total_r if temporal_score is not None else 0.0

    return {
        'xgb': w_xgb,
        'ae': w_ae,
        'gnn': w_gnn if gnn_score is not None else None,
        'temporal': w_temp if temporal_score is not None else None,
    }


# =============================================================================
# MAIN FUSION ENGINE
# =============================================================================

class FusionEngine:
    """
    Dynamic multi-model fusion engine with entropy-based confidence and
    consistency-weighted reliability calculation.

    Fusion Equation:
        S_fusion = Σ w_i × S_i  for available detectors i

        where w_i = R_i / Σ R_j, and R_i = C_i × Cons_i

        - C_i = confidence (entropy-based for probabilities)
        - Cons_i = consistency (agreement with other detectors)

    Modes:
        - "partial": Only XGB + AE available (default at startup)
        - "full": All four detectors available
    """

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD):
        """
        Initialize the FusionEngine.

        Parameters
        ----------
        threshold : float
            Decision threshold for fusion_prediction.
            fusion_score >= threshold → prediction = 1 (attack)
        """
        self.threshold = threshold
        logger.info(
            "FusionEngine initialized. threshold=%.4f",
            self.threshold
        )

    def fuse(
        self,
        xgb_score: float,
        ae_score: float,
        gnn_score: Optional[float] = None,
        temporal_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute dynamic fusion result from available component scores.

        Parameters
        ----------
        xgb_score : float
            XGBoost attack probability in [0, 1].
        ae_score : float
            Autoencoder anomaly score in [0, 1] (already normalized).
        gnn_score : float or None
            GNN malicious probability in [0, 1].
            None if not yet available.
        temporal_score : float or None
            Raw temporal score (L2 norm) or normalized [0, 1].
            If > 1.0, will be normalized.
            None if not yet available.

        Returns
        -------
        dict with keys:
            - fusion_score: float in [0, 1]
            - fusion_prediction: int (0 or 1)
            - fusion_mode: "partial" or "full"
            - weights: dict of dynamic weights
            - scores: dict of all scores used in fusion
            - available_components: list of available detector names
            - missing_components: list of unavailable detector names
            - confidence: dict of detector confidence values
            - consistency: dict of detector consistency values
            - reliability: dict of detector reliabilities
        """
        # Input validation for mandatory scores
        for name, val in [("xgb_score", xgb_score), ("ae_score", ae_score)]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"FusionEngine.fuse(): {name}={val!r} is not in [0, 1]. "
                    "All scores must be normalized before fusion."
                )

        # Validate optional scores if provided
        if gnn_score is not None and not (0.0 <= gnn_score <= 1.0):
            raise ValueError(
                f"FusionEngine.fuse(): gnn_score={gnn_score!r} is not in [0, 1]."
            )

        if temporal_score is not None:
            if temporal_score < 0:
                raise ValueError(
                    f"FusionEngine.fuse(): temporal_score={temporal_score!r} is negative."
                )
            # If temporal_score > 1, it's likely raw L2 norm - will normalize

        # Determine availability
        gnn_available = gnn_score is not None
        temporal_available = temporal_score is not None

        available_components = ['xgb', 'ae']
        missing_components = []

        if gnn_available:
            available_components.append('gnn')
        else:
            missing_components.append('gnn')

        if temporal_available:
            available_components.append('temporal')
        else:
            missing_components.append('temporal')

        # Normalize temporal score if needed
        if temporal_available and temporal_score > 1.0:
            t_normalized = normalize_temporal_score(temporal_score)
        else:
            t_normalized = temporal_score if temporal_available else None

        # Calculate dynamic weights
        weights = calculate_dynamic_weights(
            xgb_prob=xgb_score,
            ae_score=ae_score,
            gnn_score=gnn_score,
            temporal_score=t_normalized,
        )

        # Calculate confidence values for each detector
        c_xgb = calculate_confidence(xgb_score)
        c_ae = max(0.0, min(1.0, 2 * abs(ae_score - 0.5)))
        c_gnn = calculate_confidence(gnn_score) if gnn_available else None
        c_temp = max(0.0, min(1.0, 2 * abs(t_normalized - 0.5))) if temporal_available else None

        # Calculate consistency values
        cons_xgb = calculate_consistency(
            xgb_score,
            [ae_score] +
            ([gnn_score] if gnn_available else [])
        )
        cons_ae = calculate_consistency(ae_score, [xgb_score])
        cons_gnn = calculate_consistency(gnn_score, [xgb_score, ae_score]) if gnn_available else None
        cons_temp = calculate_consistency(t_normalized, [xgb_score, ae_score]) if temporal_available else None

        # Calculate reliability values
        r_xgb = calculate_reliability(c_xgb, cons_xgb)
        r_ae = calculate_reliability(c_ae, cons_ae)
        r_gnn = calculate_reliability(c_gnn, cons_gnn) if gnn_available else None
        r_temp = calculate_reliability(c_temp, cons_temp) if temporal_available else None

        # Calculate fusion score using dynamic weights
        w_xgb = weights['xgb']
        w_ae = weights['ae']
        w_gnn = weights.get('gnn', 0.0) if gnn_available else 0.0
        w_temp = weights.get('temporal', 0.0) if temporal_available else 0.0

        fusion_score = (
            w_xgb * xgb_score +
            w_ae * ae_score
        )

        if gnn_available:
            fusion_score += w_gnn * gnn_score

        if temporal_available:
            fusion_score += w_temp * t_normalized

        fusion_score = round(float(fusion_score), 6)
        fusion_prediction = 1 if fusion_score >= self.threshold else 0

        # Determine fusion mode
        fusion_mode = "full" if (gnn_available and temporal_available) else "partial"

        # Round weights
        weights_rounded = {
            k: round(v, 6) if v is not None else None
            for k, v in weights.items()
        }

        return {
            # Fusion result
            "fusion_score": fusion_score,
            "fusion_prediction": fusion_prediction,
            "fusion_mode": fusion_mode,

            # Dynamic weights
            "weights": weights_rounded,

            # Scores (normalized if needed)
            "scores": {
                "xgb": round(xgb_score, 6),
                "ae": round(ae_score, 6),
                "gnn": round(gnn_score, 6) if gnn_available else None,
                "temporal": round(t_normalized, 6) if temporal_available else None,
            },

            # Detector metadata
            "available_components": available_components,
            "missing_components": missing_components,

            # Confidence calculation transparency
            "confidence": {
                "xgb": round(c_xgb, 6),
                "ae": round(c_ae, 6),
                "gnn": round(c_gnn, 6) if gnn_available else None,
                "temporal": round(c_temp, 6) if temporal_available else None,
            },

            "consistency": {
                "xgb": round(cons_xgb, 6),
                "ae": round(cons_ae, 6),
                "gnn": round(cons_gnn, 6) if gnn_available else None,
                "temporal": round(cons_temp, 6) if temporal_available else None,
            },

            "reliability": {
                "xgb": round(r_xgb, 6),
                "ae": round(r_ae, 6),
                "gnn": round(r_gnn, 6) if gnn_available else None,
                "temporal": round(r_temp, 6) if temporal_available else None,
            },
        }

    @staticmethod
    def compute_partial_fusion(xgb_score: float, ae_score: float) -> float:
        """
        Legacy compatibility: compute partial fusion with fixed weights.
        """
        w_xgb = 0.35 / 0.55  # ≈ 0.63636
        w_ae = 0.20 / 0.55   # ≈ 0.36364
        return w_xgb * xgb_score + w_ae * ae_score