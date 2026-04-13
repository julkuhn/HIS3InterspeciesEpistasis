#!/usr/bin/env python3
"""
Compute functional-only Spearman and full-test AUC for ESM, MSA, and METL-Local 45k models.
"Functional-only" = sequences with DMS_score_bin == 1 in the test set.

For ESM/MSA:
  - X_test.npy + scaler.pkl + mlp_model.pkl -> predictions
  - y_test.npy = ground truth labels

For METL-Local 45k:
  - predictions/test_predictions.npy -> predictions
  - ground truth from splits_segmentwise_species/*_all.csv + metl_splits/*/test.txt
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score

BASE = Path("/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra")
SPLITS_BASE = BASE / "splits_segmentwise_species"
METL_SPLITS = BASE / "metl_rosetta/metl_splits"

# Segment config: (segment_key, threshold_str)
SEGMENTS = [
    ("S02", "thr0.08"),
    ("S03", "thr0.1"),
    ("S04", "thr0.08"),
    ("S05", "thr0.25"),
    ("S05_swap", "thr0.25"),
    ("S06", "thr0.18"),
    ("S07", "thr0.05"),
    ("S08", "thr0.18"),
    ("S12", "thr0.1"),
    ("S12_swap", "thr0.1"),
]

# METL-Local 45k: run_id -> (seg, thr)
METL_45K_DIR = BASE / "metl_rosetta/target_model_45k"
METL_45K_RUNS = {
    "XGWfEkoP": ("S02", "thr0.08"),
    "bat4JTEc": ("S03", "thr0.1"),
    "e74qnHH5": ("S04", "thr0.08"),
    "f6ct8j7L": ("S05", "thr0.25"),
    "F7uG4XHT": ("S06", "thr0.18"),
    "WkqJQ2ni": ("S07", "thr0.05"),
    "RCd8d3kN": ("S08", "thr0.18"),
    "N5SHB9oU": ("S12", "thr0.1"),
}


def load_ground_truth(seg, thr):
    """Load DMS scores for test set ordered by test.txt indices into all.csv."""
    seg_name = f"{seg}_{thr}"
    all_csv = SPLITS_BASE / f"super_segments_{seg_name}_all.csv"
    test_txt = METL_SPLITS / f"{seg_name}/test.txt"

    if not all_csv.exists():
        print(f"  WARNING: {all_csv} not found")
        return None, None
    if not test_txt.exists():
        print(f"  WARNING: {test_txt} not found")
        return None, None

    df_all = pd.read_csv(all_csv)
    test_idx = np.loadtxt(test_txt, dtype=int)
    df_test = df_all.iloc[test_idx].reset_index(drop=True)
    labels = df_test["DMS_score"].values
    func_mask = df_test["DMS_score_bin"].values == 1
    return labels, func_mask


def compute_metrics(y_pred, y_true, func_mask):
    """Return (spearman_full, spearman_func, auc_full)."""
    sp_full = stats.spearmanr(y_pred, y_true).statistic
    auc_full = roc_auc_score(func_mask.astype(int), y_pred)
    y_pred_func = y_pred[func_mask]
    y_true_func = y_true[func_mask]
    sp_func = stats.spearmanr(y_pred_func, y_true_func).statistic if len(y_pred_func) > 1 else float("nan")
    return sp_full, sp_func, auc_full


def process_sklearn(seg, thr, model_name, result_dir):
    """ESM / MSA: run sklearn MLP on X_test.npy using saved scaler + model."""
    result_dir = Path(result_dir)
    mlp_f = result_dir / "mlp_model.pkl"
    scaler_f = result_dir / "scaler.pkl"
    X_f = result_dir / "X_test.npy"
    y_f = result_dir / "y_test.npy"

    for p in [mlp_f, scaler_f, X_f, y_f]:
        if not p.exists():
            print(f"  [{model_name}] {seg}: missing {p.name}")
            return None

    labels, func_mask = load_ground_truth(seg, thr)
    if labels is None:
        return None

    with open(mlp_f, "rb") as f:
        model = pickle.load(f)
    with open(scaler_f, "rb") as f:
        scaler = pickle.load(f)

    X_test = np.load(X_f)
    y_true = np.load(y_f)

    # Sanity check: y_test.npy should match labels from all.csv
    if not np.allclose(y_true, labels, atol=1e-4):
        print(f"  [{model_name}] {seg}: WARNING label mismatch between y_test.npy and all.csv")

    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)

    if len(y_pred) != len(labels):
        print(f"  [{model_name}] {seg}: length mismatch preds={len(y_pred)} labels={len(labels)}")
        return None

    sp_full, sp_func, auc_full = compute_metrics(y_pred, labels, func_mask)
    n_func = func_mask.sum()
    print(f"  [{model_name}] {seg}_{thr}: n={len(labels)}, n_func={n_func} "
          f"| sp_full={sp_full:.4f} | sp_func={sp_func:.4f} | auc_full={auc_full:.4f}")
    return {
        "model": model_name, "segment": seg, "threshold": thr,
        "n_test": len(labels), "n_functional": int(n_func),
        "spearman_full": sp_full, "spearman_func_only": sp_func, "auc_full": auc_full,
    }


def process_metl45k(run_id, seg, thr):
    """METL-Local 45k: load test_predictions.npy from run dir."""
    pred_file = METL_45K_DIR / run_id / "predictions/test_predictions.npy"
    if not pred_file.exists():
        print(f"  [METL-45k] {seg}: no predictions at {pred_file}")
        return None

    labels, func_mask = load_ground_truth(seg, thr)
    if labels is None:
        return None

    y_pred = np.load(pred_file).squeeze()
    if len(y_pred) != len(labels):
        print(f"  [METL-45k] {seg}: length mismatch preds={len(y_pred)} labels={len(labels)}")
        return None

    sp_full, sp_func, auc_full = compute_metrics(y_pred, labels, func_mask)
    n_func = func_mask.sum()
    print(f"  [METL-45k] {seg}_{thr}: n={len(labels)}, n_func={n_func} "
          f"| sp_full={sp_full:.4f} | sp_func={sp_func:.4f} | auc_full={auc_full:.4f}")
    return {
        "model": "METL-Local-45k", "segment": seg, "threshold": thr,
        "n_test": len(labels), "n_functional": int(n_func),
        "spearman_full": sp_full, "spearman_func_only": sp_func, "auc_full": auc_full,
    }


results = []

print("=" * 70)
print("ESM results")
print("=" * 70)
for seg, thr in SEGMENTS:
    r = process_sklearn(seg, thr, "ESM", BASE / f"esm_results/{seg}")
    if r:
        results.append(r)

print()
print("=" * 70)
print("MSA results")
print("=" * 70)
for seg, thr in SEGMENTS:
    r = process_sklearn(seg, thr, "MSA", BASE / f"msa_results/{seg}")
    if r:
        results.append(r)

print()
print("=" * 70)
print("METL-Local 45k results")
print("=" * 70)
for run_id, (seg, thr) in METL_45K_RUNS.items():
    r = process_metl45k(run_id, seg, thr)
    if r:
        results.append(r)

# Save results
out_df = pd.DataFrame(results)
out_path = BASE / "metl_rosetta/functional_metrics.csv"
out_df.to_csv(out_path, index=False)
print()
print(f"Saved to {out_path}")
print()
print(out_df.to_string(index=False))
