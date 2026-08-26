"""
config.py - Central configuration for the NIDS hybrid architecture.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Person 2+3 model artifacts
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.pkl")
AE_MODEL_PATH = os.path.join(MODELS_DIR, "ae_model.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
AE_ARTIFACTS_PATH = os.path.join(MODELS_DIR, "artifacts.json")
XGB_ARTIFACTS_PATH = os.path.join(MODELS_DIR, "xgb_artifacts.json")
FUSION_CONFIG_PATH = os.path.join(MODELS_DIR, "fusion_config.json")

# Person 1 GNN artifacts
GNN_MODEL_PATH = os.path.join(MODELS_DIR, "dynamic_temporal_gnn.pt")
GNN_SCALERS_PATH = os.path.join(MODELS_DIR, "gnn_scalers.pkl")

# Dashboard
DASHBOARD_PORT = 8051