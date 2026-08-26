"""
modules/feature_schema_validator.py
====================================
LIVE TRAFFIC FEATURE SCHEMA VALIDATOR

Validates that live packet flow capture engines produce EXACTLY the same 49 NetFlow/IPFIX
feature columns in the exact order expected by the trained XGBoost, Autoencoder, and GNN models.

Validation Result Contract:
  {
      "valid": bool,               # True ONLY when missing_features is empty
      "missing_features": list,    # features absent from the input dict
      "zero_filled_features": list,# features present but NaN/Inf → zeroed
      "unexpected_features": list, # input keys not in the 49-feature schema
      "feature_count": int,        # number of features in output vector
      "feature_vector": np.ndarray # final 49-feature float32 vector
  }

IMPORTANT: missing_features being empty is the ONLY condition for valid=True.
A flow that has all 49 keys present, even if some are zero, is valid.
A flow that is missing any key is invalid — even though zero-filling still occurs
for model compatibility, the validation result correctly reports the gap.
"""

import os
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(HERE, "models")
ARTIFACTS_PATH = os.path.join(MODELS_DIR, "xgb_artifacts.json")


from feature_schema import FEATURE_COLS


class FeatureSchemaValidator:
    """
    Validates Live Traffic Capture Feature Schema against the 49-feature Offline Model Schema.

    Strict contract:
      - valid=True if and only if ALL 49 expected features are present in the input.
      - Missing features are zero-filled for model compatibility but are ALWAYS reported.
      - NaN/Inf values are zeroed and reported in zero_filled_features.
      - Extra (unexpected) keys in the input are reported but do NOT affect validity.
    """

    def __init__(self):
        self.expected_feature_cols = FEATURE_COLS
        self.expected_num_features = len(FEATURE_COLS)
        self._expected_set = set(FEATURE_COLS)

    def validate_schema(self, live_flow_record):
        """
        Validates a live flow record dictionary or feature vector.

        Args:
            live_flow_record: dict mapping feature names to values,
                              or list/ndarray of 49 float values.

        Returns:
            dict with keys:
                valid              : bool — True only when missing_features is empty
                missing_features   : List[str] — feature keys absent from input dict
                zero_filled_features: List[str] — keys present but NaN/Inf → 0
                unexpected_features: List[str] — keys in input not in schema
                feature_count      : int — length of output vector
                feature_vector     : np.ndarray of shape (49,) dtype float32
        """
        missing_features = []
        zero_filled_features = []
        unexpected_features = []

        if isinstance(live_flow_record, dict):
            # Check for unexpected keys (excluding metadata keys starting with '_')
            input_keys = {k for k in live_flow_record if not k.startswith("_")}
            unexpected_features = sorted(input_keys - self._expected_set)

            feat_vals = []
            for col in self.expected_feature_cols:
                if col not in live_flow_record:
                    missing_features.append(col)
                    feat_vals.append(0.0)  # zero-fill for model compatibility
                else:
                    try:
                        val = float(live_flow_record[col])
                        if np.isnan(val) or np.isinf(val):
                            zero_filled_features.append(col)
                            val = 0.0
                        feat_vals.append(val)
                    except (ValueError, TypeError):
                        missing_features.append(col)  # treat unparseable as missing
                        feat_vals.append(0.0)

            feat_arr = np.array(feat_vals, dtype=np.float32)

        elif isinstance(live_flow_record, (list, np.ndarray)):
            feat_arr = np.array(live_flow_record, dtype=np.float32)
            if len(feat_arr) != self.expected_num_features:
                # Treat length mismatch as all-features missing
                missing_features = [f"<length_mismatch: got {len(feat_arr)} expected {self.expected_num_features}>"]
            # Replace NaN/Inf
            bad_mask = ~np.isfinite(feat_arr)
            if bad_mask.any():
                for i in np.where(bad_mask)[0]:
                    if i < len(self.expected_feature_cols):
                        zero_filled_features.append(self.expected_feature_cols[i])
                feat_arr = np.where(bad_mask, 0.0, feat_arr).astype(np.float32)
        else:
            return {
                "valid": False,
                "missing_features": [f"Invalid input type: {type(live_flow_record)}"],
                "zero_filled_features": [],
                "unexpected_features": [],
                "feature_count": 0,
                "feature_vector": None,
            }

        if missing_features:
            logger.warning(
                "Feature schema validation FAILED: %d missing features: %s",
                len(missing_features),
                missing_features[:10],  # log first 10
            )
        if zero_filled_features:
            logger.debug(
                "Feature schema: %d features zero-filled (NaN/Inf): %s",
                len(zero_filled_features),
                zero_filled_features,
            )

        return {
            "valid": len(missing_features) == 0,
            "missing_features": missing_features,
            "zero_filled_features": zero_filled_features,
            "unexpected_features": unexpected_features,
            "feature_count": len(feat_arr),
            "feature_vector": feat_arr,
        }

    def extract_operational_fields(self, live_flow_dict, model_risk=0.0, attack_path=None, xai_exp=None):
        """
        Extracts the 11 core operational fields required for SOC Dashboard display.
        """
        src_ip   = str(live_flow_dict.get("IPV4_SRC_ADDR") or live_flow_dict.get("_src_ip", "0.0.0.0"))
        dst_ip   = str(live_flow_dict.get("IPV4_DST_ADDR") or live_flow_dict.get("_dst_ip", "0.0.0.0"))
        dst_port = int(live_flow_dict.get("L4_DST_PORT", 0))
        protocol = int(live_flow_dict.get("PROTOCOL", 0))
        byte_cnt = float(live_flow_dict.get("IN_BYTES", 0.0))
        pkt_cnt  = float(live_flow_dict.get("IN_PKTS", 0.0))
        duration = float(live_flow_dict.get("FLOW_DURATION_MILLISECONDS", 0.0))

        level = "Level 1 — Normal"
        if model_risk >= 0.80:
            level = "Level 3 — High Risk"
        elif model_risk >= 0.50:
            level = "Level 2 — Suspicious"

        return {
            "source_host":      f"HOST:{src_ip}",
            "destination_host": f"HOST:{dst_ip}",
            "service_port":     f"SERVICE:{dst_port}/{protocol}",
            "protocol":         protocol,
            "bytes":            byte_cnt,
            "packets":          pkt_cnt,
            "duration":         duration,
            "model_risk":       float(model_risk),
            "attack_path":      attack_path or [f"HOST:{src_ip}", f"SERVICE:{dst_port}/{protocol}", f"HOST:{dst_ip}"],
            "xai_explanation":  xai_exp or {},
            "level":            level,
        }


if __name__ == "__main__":
    validator = FeatureSchemaValidator()
    print(f"Feature Schema Validator initialized.")
    print(f"Expected Feature Count: {validator.expected_num_features}")
    print(f"Feature Schema Sample: {validator.expected_feature_cols[:5]} ...")

    # Self-test: full feature dict
    full_flow = {col: 1.0 for col in validator.expected_feature_cols}
    result = validator.validate_schema(full_flow)
    assert result["valid"] is True
    assert result["missing_features"] == []
    print(f"Self-test (full dict): valid={result['valid']}, missing={result['missing_features']}")

    # Self-test: partial feature dict
    partial_flow = {col: 1.0 for col in validator.expected_feature_cols[:10]}
    result2 = validator.validate_schema(partial_flow)
    assert result2["valid"] is False
    assert len(result2["missing_features"]) == 39
    print(f"Self-test (partial dict): valid={result2['valid']}, missing_count={len(result2['missing_features'])}")
    print("All self-tests passed.")
