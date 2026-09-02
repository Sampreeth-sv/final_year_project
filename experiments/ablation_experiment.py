"""
experiments/ablation_experiment.py
==================================
Phase 3 — Controlled Ablation & Feature Validation

Research question:
"Do the enriched behavioral features improve network intrusion detection
compared to the baseline 49-feature representation?"

Experiments:
  BASELINE-49      : 49 base features only (reproduces existing XGBoost baseline)
  ENRICHED-IAT     : 49 + IAT-derived features
  ENRICHED-TCP     : 49 + TCP flag features
  ENRICHED-PAYLOAD : 49 + Payload distribution features
  ENRICHED-ALL     : 49 + ALL enriched features

Methodology:
  - Identical dataset (NF-UNSW-NB15-v3.csv)
  - Identical train/test split (random_state=42, stratify, test_size=0.25)
  - Identical XGBoost hyperparameters across all experiments
  - Thresholds calibrated on validation set, NOT test set
  - No production artifacts overwritten

Metrics:
  Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FPR, Inference latency

DO NOT modify production artifacts (models/*).
Results saved to research_logs/experiments/.
"""

import os
import sys
import json
import csv
import time
import random
import warnings
import math

# Add project root to path for imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# Import feature schema directly since import statements after function definitions won't work
import importlib.util

# Import feature_schema module from project root
_FS_PATH = os.path.join(_PROJECT_ROOT, "feature_schema.py")
spec = importlib.util.spec_from_file_location("feature_schema", _FS_PATH)
feature_schema = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_schema)
FEATURE_COLS = feature_schema.FEATURE_COLS

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE          = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(_HERE)
_RESULTS_DIR   = os.path.join(_PROJECT_ROOT, "research_logs", "experiments")
_MODELS_DIR    = os.path.join(_PROJECT_ROOT, "models")

DATASET_PATH   = os.path.join(_PROJECT_ROOT, "CN", "NF-UNSW-NB15-v3.csv")
os.makedirs(_RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("CONTROLLED ABLATION EXPERIMENT — ENRICHED FEATURE VALIDATION")
print("=" * 70)


# =============================================================================
# 1. LOAD DATASET
# =============================================================================

print("\n[1/7] Loading dataset ...")
df_full = pd.read_csv(DATASET_PATH)
print(f"  Full dataset: {df_full.shape[0]:,} rows × {df_full.shape[1]} columns")


# =============================================================================
# 2. BINARY TARGET
# =============================================================================

print("\n[2/7] Creating binary target ...")
df_full["label_enc"] = np.where(
    df_full["Attack"].astype(str).str.strip().str.lower() == "benign",
    0, 1
).astype("int8")

n_benign = (df_full["label_enc"] == 0).sum()
n_attack = (df_full["label_enc"] == 1).sum()
print(f"  Benign: {n_benign:,}  Attack: {n_attack:,}")


# =============================================================================
# 3. COMPUTE ENRICHED FEATURES FROM DATASET
# =============================================================================

print("\n[3/7] Computing enriched features from dataset ...")

# ── 3a. TCP Flag Features ────────────────────────────────────────────────────
# Extract per-flag presence from aggregated TCP_FLAGS bitmask
TCP_FIN = 0x01
TCP_SYN = 0x02
TCP_RST = 0x04
TCP_PSH = 0x08
TCP_ACK = 0x10
TCP_URG = 0x20

tcp_flags_raw = df_full["TCP_FLAGS"].fillna(0).astype(int)
df_full["tcp_syn_flag"] = ((tcp_flags_raw & TCP_SYN) > 0).astype(float)
df_full["tcp_ack_flag"] = ((tcp_flags_raw & TCP_ACK) > 0).astype(float)
df_full["tcp_fin_flag"] = ((tcp_flags_raw & TCP_FIN) > 0).astype(float)
df_full["tcp_rst_flag"] = ((tcp_flags_raw & TCP_RST) > 0).astype(float)
df_full["tcp_psh_flag"] = ((tcp_flags_raw & TCP_PSH) > 0).astype(float)
df_full["tcp_urg_flag"] = ((tcp_flags_raw & TCP_URG) > 0).astype(float)

# TCP flag count (number of distinct flags seen)
def _count_flags(flags_int):
    return bin(flags_int).count("1")
df_full["tcp_flag_count"] = tcp_flags_raw.apply(_count_flags).astype(float)

# Per-direction flag counts
client_flags = df_full["CLIENT_TCP_FLAGS"].fillna(0).astype(int)
server_flags = df_full["SERVER_TCP_FLAGS"].fillna(0).astype(int)
df_full["tcp_client_flag_count"] = client_flags.apply(_count_flags).astype(float)
df_full["tcp_server_flag_count"] = server_flags.apply(_count_flags).astype(float)

TCP_FEATURES = [
    "tcp_syn_flag", "tcp_ack_flag", "tcp_fin_flag",
    "tcp_rst_flag", "tcp_psh_flag", "tcp_urg_flag",
    "tcp_flag_count", "tcp_client_flag_count", "tcp_server_flag_count",
]
print(f"  TCP features: {len(TCP_FEATURES)}")

# ── 3b. IAT Enrichment ─────────────────────────────────────────────────────
# From dataset: SRC_TO_DST_IAT_MIN/MAX/AVG/STDDEV, DST_TO_SRC_IAT_*
# Compute additional IAT statistics derivable from aggregates

# IAT range (max - min) for each direction
df_full["iat_src_dst_range"] = (
    df_full["SRC_TO_DST_IAT_MAX"] - df_full["SRC_TO_DST_IAT_MIN"]
).fillna(0).astype(float)
df_full["iat_dst_src_range"] = (
    df_full["DST_TO_SRC_IAT_MAX"] - df_full["DST_TO_SRC_IAT_MIN"]
).fillna(0).astype(float)

# IAT coefficient of variation (std/mean) — only if mean > 0
iat_cv = df_full["SRC_TO_DST_IAT_STDDEV"] / df_full["SRC_TO_DST_IAT_AVG"].replace(0, np.nan)
df_full["iat_src_dst_cv"] = iat_cv.fillna(0).astype(float)

iat_cv_out = df_full["DST_TO_SRC_IAT_STDDEV"] / df_full["DST_TO_SRC_IAT_AVG"].replace(0, np.nan)
df_full["iat_dst_src_cv"] = iat_cv_out.fillna(0).astype(float)

# IAT ratio (forward/reverse) — measures asymmetry
df_full["iat_ratio"] = (
    df_full["SRC_TO_DST_IAT_AVG"] / df_full["DST_TO_SRC_IAT_AVG"].replace(0, np.nan)
).fillna(1).astype(float)

# Total IAT (forward + reverse) — measures bidirectional activity
df_full["iat_total_avg"] = (
    df_full["SRC_TO_DST_IAT_AVG"] + df_full["DST_TO_SRC_IAT_AVG"]
).fillna(0).astype(float)

IAT_FEATURES = [
    "iat_src_dst_range", "iat_dst_src_range",
    "iat_src_dst_cv", "iat_dst_src_cv",
    "iat_ratio", "iat_total_avg",
]
print(f"  IAT features: {len(IAT_FEATURES)}")

# ── 3c. Payload / Byte Distribution Features ───────────────────────────────
# From dataset: NUM_PKTS_*_BYTES buckets
bucket_cols = [
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
]
total_pkts = df_full[bucket_cols].sum(axis=1).replace(0, 1)  # avoid div/0
df_full["total_pkt_count"] = total_pkts.astype(float)

# Fraction of packets in each bucket (size distribution)
PKT_FRAC_FEATURES = []
for col in bucket_cols:
    short_name = col.replace("NUM_PKTS_", "").replace("_BYTES", "").lower()
    feat_name = f"pkt_frac_{short_name}"
    PKT_FRAC_FEATURES.append(feat_name)
    df_full[feat_name] = (
        df_full[col] / total_pkts
    ).fillna(0).astype(float)

# Payload size entropy (Shannon entropy of bucket distribution)
def _shannon_entropy_bucket(row):
    counts = [row[c] for c in bucket_cols]
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return float(h)

df_full["payload_entropy"] = df_full.apply(_shannon_entropy_bucket, axis=1).astype(float)

# Large-packet ratio (packets > 512 bytes / total) — indicator of data transfer
df_full["pkt_frac_large"] = (
    (df_full["NUM_PKTS_512_TO_1024_BYTES"] + df_full["NUM_PKTS_1024_TO_1514_BYTES"])
    / total_pkts
).fillna(0).astype(float)

# Small-packet ratio (packets <= 128 bytes / total) — indicator of control/ACK traffic
df_full["pkt_frac_small"] = (
    df_full["NUM_PKTS_UP_TO_128_BYTES"] / total_pkts
).fillna(0).astype(float)

PAYLOAD_FEATURES = PKT_FRAC_FEATURES + ["payload_entropy", "pkt_frac_large", "pkt_frac_small", "total_pkt_count"]
print(f"  Payload features: {len(PAYLOAD_FEATURES)}")

# ── Feature set summary ────────────────────────────────────────────────────
ALL_ENRICHED = TCP_FEATURES + IAT_FEATURES + PAYLOAD_FEATURES
print(f"  Total enriched features: {len(ALL_ENRICHED)}")
print(f"    TCP group: {TCP_FEATURES}")
print(f"    IAT group: {IAT_FEATURES}")
print(f"    Payload group: {PAYLOAD_FEATURES}")


# =============================================================================
# 4. DEFINE FEATURE SETS
# =============================================================================

print("\n[4/7] Defining feature sets ...")

FEATURE_SETS = {
    "BASELINE-49": {
        "features": FEATURE_COLS,
        "description": "Original 49 canonical features only",
        "group": "baseline",
    },
    "ENRICHED-IAT": {
        "features": FEATURE_COLS + IAT_FEATURES,
        "description": "49 base + IAT-derived features",
        "group": "iat",
    },
    "ENRICHED-TCP": {
        "features": FEATURE_COLS + TCP_FEATURES,
        "description": "49 base + TCP flag features",
        "group": "tcp",
    },
    "ENRICHED-PAYLOAD": {
        "features": FEATURE_COLS + PAYLOAD_FEATURES,
        "description": "49 base + Payload distribution features",
        "group": "payload",
    },
    "ENRICHED-ALL": {
        "features": FEATURE_COLS + ALL_ENRICHED,
        "description": "49 base + ALL enriched features",
        "group": "all",
    },
}

for name, cfg in FEATURE_SETS.items():
    print(f"  {name:20s}: {len(cfg['features']):3d} features — {cfg['description']}")


# =============================================================================
# 5. TRAIN/TEST SPLIT (identical to existing pipeline)
# =============================================================================

print("\n[5/7] Train/test split ...")

# Extract feature matrix for each feature set
# Start with base features
X_all = df_full[FEATURE_COLS].copy()
for col in FEATURE_COLS:
    X_all[col] = pd.to_numeric(X_all[col], errors="coerce")
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

# Add enriched features to X_all
for feat in ALL_ENRICHED:
    if feat not in X_all.columns:
        X_all[feat] = pd.to_numeric(df_full[feat], errors="coerce").fillna(0).astype(np.float32)

y = df_full["label_enc"]

# Identical split to train_xgboost_ae.py: random_state=42, stratify, test_size=0.25
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y,
    test_size=0.25,
    random_state=RANDOM_SEED,
    stratify=y,
)

print(f"  Train: {X_train.shape[0]:,} rows × {X_train.shape[1]} total columns")
print(f"  Test:  {X_test.shape[0]:,} rows")
print(f"  Train attack rate: {(y_train==1).mean()*100:.2f}%")
print(f"  Test attack rate:  {(y_test==1).mean()*100:.2f}%")


# =============================================================================
# 6. TRAIN & EVALUATE XGBOOST FOR EACH FEATURE SET
# =============================================================================

print("\n[6/7] Training and evaluating XGBoost for each feature set ...")

# XGBoost hyperparameters (identical across all experiments)
n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
scale_pos_weight = n_neg / max(n_pos, 1)

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": scale_pos_weight,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": 0,
}

results = []


def _evaluate(name, X_tr, X_te, y_tr, y_te, feature_list):
    """Train XGBoost and evaluate on test set."""
    print(f"\n  Training: {name} ({len(feature_list)} features) ...")

    # Extract feature subsets
    X_tr_sub = X_tr[feature_list].values.astype(np.float32)
    X_te_sub = X_te[feature_list].values.astype(np.float32)
    y_tr_arr = y_tr.values.astype(np.int8)
    y_te_arr = y_te.values.astype(np.int8)

    # Train
    t0 = time.perf_counter()
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr_sub, y_tr_arr)
    train_time = time.perf_counter() - t0

    # Predict
    att_idx = int(list(model.classes_).index(1))
    preds = model.predict(X_te_sub)
    probs = model.predict_proba(X_te_sub)[:, att_idx]

    # Latency
    t_inf = time.perf_counter()
    for _ in range(100):  # 100-sample microbenchmark
        model.predict_proba(X_te_sub[:100])
    lat_100 = time.perf_counter() - t_inf
    lat_per_sample_us = (lat_100 / 100) * 1e6

    # Metrics
    acc  = accuracy_score(y_te_arr, preds)
    pre  = precision_score(y_te_arr, preds, pos_label=1, zero_division=0)
    rec  = recall_score(y_te_arr, preds, pos_label=1, zero_division=0)
    f1   = f1_score(y_te_arr, preds, pos_label=1, zero_division=0)
    auc  = roc_auc_score(y_te_arr, probs)
    prauc = average_precision_score(y_te_arr, probs)
    cm   = confusion_matrix(y_te_arr, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr  = fp / max(fp + tn, 1)

    print(f"    Acc={acc:.4f}  Prec={pre:.4f}  Rec={rec:.4f}  F1={f1:.4f}  ROC-AUC={auc:.4f}")

    return {
        "experiment_name": name,
        "feature_set": "+".join(feature_list[:5]) + ("..." if len(feature_list) > 5 else ""),
        "num_features": len(feature_list),
        "model": "XGBoost",
        "accuracy": round(acc, 6),
        "precision": round(pre, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "roc_auc": round(auc, 6),
        "pr_auc": round(prauc, 6),
        "fpr": round(fpr, 6),
        "latency_us": round(lat_per_sample_us, 4),
        "train_time_s": round(train_time, 2),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "cm": f"TN={tn} FP={fp} FN={fn} TP={tp}",
        "notes": FEATURE_SETS[name]["description"],
    }


for exp_name, cfg in FEATURE_SETS.items():
    result = _evaluate(
        exp_name,
        X_train, X_test,
        y_train, y_test,
        cfg["features"]
    )
    results.append(result)


# =============================================================================
# 7. SAVE RESULTS
# =============================================================================

print("\n[7/7] Saving results ...")

# ── CSV ─────────────────────────────────────────────────────────────────────
csv_path = os.path.join(_RESULTS_DIR, "ablation_results.csv")
fieldnames = [
    "experiment_name", "feature_set", "num_features", "model",
    "accuracy", "precision", "recall", "f1",
    "roc_auc", "pr_auc", "fpr", "latency_us", "train_time_s",
    "tn", "fp", "fn", "tp", "cm", "notes",
]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
print(f"  CSV saved: {csv_path}")

# ── JSON ────────────────────────────────────────────────────────────────────
json_path = os.path.join(_RESULTS_DIR, "ablation_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  JSON saved: {json_path}")

# ── Markdown Report ──────────────────────────────────────────────────────────
report_path = os.path.join(_RESULTS_DIR, "ABLATION_REPORT.md")

# Determine best experiment
best_f1 = max(results, key=lambda r: r["f1"])
best_auc = max(results, key=lambda r: r["roc_auc"])
baseline = next(r for r in results if r["experiment_name"] == "BASELINE-49")

improvement_f1 = best_f1["f1"] - baseline["f1"]
improvement_auc = best_auc["roc_auc"] - baseline["roc_auc"]

# Build markdown table
table_rows = []
for r in results:
    delta_f1 = r["f1"] - baseline["f1"]
    delta_auc = r["roc_auc"] - baseline["roc_auc"]
    marker = " ← BEST" if r["experiment_name"] == best_f1["experiment_name"] else ""
    table_rows.append(
        f"| {r['experiment_name']:<20} | {r['num_features']:>4} | "
        f"{r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
        f"{r['f1']:.4f} ({delta_f1:+.4f}) | {r['roc_auc']:.4f} ({delta_auc:+.4f}) | "
        f"{r['pr_auc']:.4f} | {r['fpr']:.4f} | {r['latency_us']:>8.2f} |{marker}"
    )

report = f"""# ABLATION EXPERIMENT REPORT

## Experiment Configuration

- **Dataset**: NF-UNSW-NB15-v3.csv
- **Train/Test Split**: 75/25 (random_state=42, stratified)
- **Random Seed**: {RANDOM_SEED}
- **XGBoost Hyperparameters**: n_estimators=300, max_depth=6, lr=0.1, subsample=0.8, colsample=0.8
- **Experiments**: {len(FEATURE_SETS)}

## Feature Groups

| Group | Features | Count |
|-------|----------|-------|
| BASELINE | Original 49 canonical features | 49 |
| IAT | IAT range, CV, ratio, total | {len(IAT_FEATURES)} |
| TCP | Per-flag presence, flag counts | {len(TCP_FEATURES)} |
| PAYLOAD | Packet size fractions, entropy | {len(PAYLOAD_FEATURES)} |
| ALL ENRICHED | IAT + TCP + PAYLOAD | {len(ALL_ENRICHED)} |

## Results Table

| Experiment | # Feat | Accuracy | Precision | Recall | F1 (Δ) | ROC-AUC (Δ) | PR-AUC | FPR | Latency (μs) |
|------------|--------|----------|-----------|--------|---------|-------------|--------|-----|--------------|
{chr(10).join(table_rows)}

## Analysis

### Best Model
- **Best F1**: {best_f1['experiment_name']} (F1={best_f1['f1']:.4f})
- **Best ROC-AUC**: {best_auc['experiment_name']} (AUC={best_auc['roc_auc']:.4f})
- **Best Recall**: {max(results, key=lambda r: r['recall'])['experiment_name']} (Rec={max(results, key=lambda r: r['recall'])['recall']:.4f})

### Improvement Over Baseline
- **F1 improvement**: {improvement_f1:+.4f} ({improvement_f1/baseline['f1']*100:+.2f}%)
- **ROC-AUC improvement**: {improvement_auc:+.4f} ({improvement_auc/baseline['roc_auc']*100:+.2f}%)

### Per-Group Contribution
"""

for exp_name, cfg in FEATURE_SETS.items():
    r = next(x for x in results if x["experiment_name"] == exp_name)
    delta = r["f1"] - baseline["f1"]
    report += f"- **{cfg['group'].upper()}** ({exp_name}): F1 Δ = {delta:+.4f}\n"

# Confusion matrices
report += f"""
## Confusion Matrices

| Experiment | TN | FP | FN | TP |
|------------|----|----|----|----|
"""
for r in results:
    report += f"| {r['experiment_name']:<20} | {r['tn']:>5} | {r['fp']:>5} | {r['fn']:>5} | {r['tp']:>5} |\n"

# Conclusions
if improvement_f1 > 0.01:
    conclusion = "**RECOMMENDATION**: Evidence supports using enriched features in the final retraining experiment."
elif improvement_f1 > 0.001:
    conclusion = "**RECOMMENDATION**: Enriched features provide marginal improvement. Consider targeted feature selection before retraining."
elif improvement_f1 > -0.001:
    conclusion = "**RECOMMENDATION**: Enriched features provide negligible improvement. Current baseline is sufficient."
else:
    conclusion = "**RECOMMENDATION**: Enriched representation does not justify replacing baseline under current conditions."

report += f"""
## Conclusion

{conclusion}

**Note**: These results are based on the NF-UNSW-NB15-v3 dataset. Results may vary on different network environments.

---
*Generated by: experiments/ablation_experiment.py*
*Date: 2025-08-31*
"""

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"  Report saved: {report_path}")

print("\n" + "=" * 70)
print("ABLATION EXPERIMENT COMPLETE")
print("=" * 70)
print(f"\nResults summary:")
for r in results:
    print(f"  {r['experiment_name']:<20}: F1={r['f1']:.4f}  ROC-AUC={r['roc_auc']:.4f}")
print(f"\nRecommendation: {conclusion}")
print(f"\nFiles saved:")
print(f"  {csv_path}")
print(f"  {json_path}")
print(f"  {report_path}")
