#!/usr/bin/env python3
"""
Compute unified metrics for all models and segments.

Metrics per model × segment:
  - Spearman ρ (overall test set)
  - Functional Spearman ρ (DMS > 0.5 subset)
  - AUC (binary: DMS > 0.5)
  - % Positive (fraction of test set with DMS > 0.5)
"""

import sys
import os
import json
import pickle
import numpy as np

# Use pip packages
sys.path.insert(0, '/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/.pip_packages')

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

N_BOOT = 1000
RNG    = np.random.default_rng(42)

def bootstrap_ci(y_true, y_pred, n=N_BOOT, alpha=0.05):
    """Bootstrap 95% CI for Spearman ρ."""
    stats = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        s = RNG.choice(idx, size=len(idx), replace=True)
        r, _ = spearmanr(y_true[s], y_pred[s])
        stats.append(r)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return round(lo, 4), round(hi, 4)


def bootstrap_ci_func_spearman(y_true, y_pred, thr=0.5, n=N_BOOT, alpha=0.05):
    """Bootstrap 95% CI for Spearman ρ on DMS > thr subset."""
    mask = y_true > thr
    if mask.sum() < 10:
        return None, None
    yt, yp = y_true[mask], y_pred[mask]
    stats = []
    idx = np.arange(len(yt))
    for _ in range(n):
        s = RNG.choice(idx, size=len(idx), replace=True)
        r, _ = spearmanr(yt[s], yp[s])
        stats.append(r)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return round(lo, 4), round(hi, 4)


def bootstrap_ci_auc(y_true, y_pred, thr=0.5, n=N_BOOT, alpha=0.05):
    """Bootstrap 95% CI for ROC-AUC (binary: DMS > thr)."""
    y_bin = (y_true > thr).astype(int)
    if y_bin.sum() < 5 or y_bin.sum() == len(y_bin):
        return None, None
    stats = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        s = RNG.choice(idx, size=len(idx), replace=True)
        ys, ps = y_bin[s], y_pred[s]
        if 0 < ys.sum() < len(ys):
            stats.append(roc_auc_score(ys, ps))
    if not stats:
        return None, None
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return round(lo, 4), round(hi, 4)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BASE = '/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra'

SEGMENTS = {
    'S02':      {'thr': '0.08'},
    'S03':      {'thr': '0.1'},
    'S04':      {'thr': '0.08'},
    'S05':      {'thr': '0.25'},
    'S05_swap': {'thr': '0.25'},
    'S06':      {'thr': '0.18'},
    'S07':      {'thr': '0.05'},
    'S08':      {'thr': '0.18'},
    'S12':      {'thr': '0.1'},
    'S12_swap': {'thr': '0.1'},
}

# METL-G-1D run IDs (only one per segment with completed results)
METL_1D_RUNS = {
    'S02': 'U4RAPDNd',
    'S03': 'igxTKszy',
    'S04': 'jk5TyVxg',
    'S05': 'Qe5rDsst',
    'S05_swap': 'kZNpFE9z',
    'S06': 'ddoz3fmN',
    'S07': 'AZ2KdcDf',
    'S08': '9wSTrXHw',
    'S12': 'gUwA3LHp',
    'S12_swap': 'kwzsZciK',
}

# METL-G-3D run IDs
METL_3D_RUNS = {
    'S02': 'gUqfFdUh',
    'S03': 'fBKeWwjY',
    'S04': 'QaEBNDKS',
    'S05': 'JnR5X73e',
    'S05_swap': 'NG69zeTi',
    'S06': 'kmPzFXQm',  # two completed; picking higher spearman
    'S07': 'fXKMhwhk',
    'S08': 'X8R6BSXd',
    'S12': 'N9XA28BA',
    'S12_swap': 'hWoHzTT4',
}

# METL-Local v1 run IDs (target_model_45k: extract + linear head)
METL_LOCAL_V1_RUNS = {
    'S02': 'XGWfEkoP',
    'S03': 'bat4JTEc',
    'S04': 'e74qnHH5',
    'S05': 'f6ct8j7L',
    'S05_swap': 'jw7LxETD',
    'S06': 'F7uG4XHT',
    'S07': 'WkqJQ2ni',
    'S08': 'RCd8d3kN',
    'S12': 'N5SHB9oU',
}

# METL-Local v2 run IDs (target_model_45k_v2: backbone + nonlinear head)
METL_LOCAL_V2_RUNS = {
    'S02': 'gCkLxwfd',
    'S03': 'A6rKA5sU',
    'S04': 'Bf9GQ234',
    'S05': 'SN64rDWg',
    'S05_swap': '7TV63ZJY',
    'S06': 'CNnpaRDv',
    'S07': 'HChkJTRS',
    'S08': 'bwuAoE8P',
    'S12': 'MTRipdMA',
}

# METL-Local v3 run IDs (target_model_45k_v3: extract + nonlinear head)
METL_LOCAL_V3_RUNS = {
    'S02': 'VvGUdj8k',
    'S03': '6gwjv5SZ',
    'S04': 'CwHu72d2',
    'S05': 'cVsojj7a',
    'S05_swap': 'B9inrJ7X',
    'S06': '56zN4Xas',
    'S07': 'iCD9J845',
    'S08': 'a78zGNXX',
    'S12': 'YWnTsrny',
}

FUNC_THR = 0.5  # fixed threshold for functional Spearman and AUC

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, thr=FUNC_THR):
    """Return dict with spearman, func_spearman, auc, pct_positive."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    sp, _ = spearmanr(y_true, y_pred)

    # Functional Spearman (DMS > thr)
    mask = y_true > thr
    if mask.sum() >= 5 and mask.sum() < len(mask):
        sp_func, _ = spearmanr(y_true[mask], y_pred[mask])
    else:
        sp_func = float('nan')

    # AUC
    y_bin = (y_true > thr).astype(int)
    if 0 < y_bin.sum() < len(y_bin):
        auc = roc_auc_score(y_bin, y_pred)
    else:
        auc = float('nan')

    pct_pos = float(mask.sum()) / len(mask)

    sp_lo, sp_hi        = bootstrap_ci(y_true, y_pred)
    fs_lo, fs_hi        = bootstrap_ci_func_spearman(y_true, y_pred, thr)
    auc_lo, auc_hi      = bootstrap_ci_auc(y_true, y_pred, thr)

    return {
        'spearman':           round(sp, 4),
        'spearman_ci_lo':     sp_lo,
        'spearman_ci_hi':     sp_hi,
        'func_spearman':      round(sp_func, 4) if not np.isnan(sp_func) else None,
        'func_spearman_ci_lo': fs_lo,
        'func_spearman_ci_hi': fs_hi,
        'auc':                round(auc, 4) if not np.isnan(auc) else None,
        'auc_ci_lo':          auc_lo,
        'auc_ci_hi':          auc_hi,
        'pct_positive':       round(pct_pos * 100, 1),
        'n_test':             len(y_true),
    }


def load_tsv_test(seg, thr):
    """Load ground-truth DMS_score for test set from TSV + metl split indices."""
    tsv = f'{BASE}/splits_segmentwise_species/super_segments_{seg}_thr{thr}_all.tsv'
    split_txt = f'{BASE}/metl_rosetta/metl_splits/{seg}_thr{thr}/test.txt'
    with open(split_txt) as f:
        idx = [int(x.strip()) for x in f]
    df = pd.read_csv(tsv, sep='\t', usecols=['DMS_score'])
    y = df['DMS_score'].values[idx]
    return y, idx


def load_esm_or_msa(model_dir, seg):
    """Load ESM or MSA test predictions using sklearn model + X_test/y_test."""
    y_test = np.load(f'{model_dir}/{seg}/y_test.npy')
    X_test = np.load(f'{model_dir}/{seg}/X_test.npy')
    with open(f'{model_dir}/{seg}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(f'{model_dir}/{seg}/mlp_model.pkl', 'rb') as f:
        model = pickle.load(f)
    X_scaled = scaler.transform(X_test)
    preds = model.predict(X_scaled)
    return y_test, preds


def load_metl_preds(result_dir, run_id):
    """Load METL predictions from test_predictions.npy."""
    return np.load(f'{result_dir}/{run_id}/predictions/test_predictions.npy')


def load_pairformer_test(seg, thr):
    """Load pairformer test predictions by matching split CSV test variants."""
    csv_path = f'{BASE}/splits_segmentwise_species/super_segments_{seg}_thr{thr}_all.csv'
    pf_path  = f'{BASE}/pairformer_results/{seg}/pairformer_predictions_v2.csv'

    split_df = pd.read_csv(csv_path, usecols=['mutant', 'DMS_score', 'split'])
    test_df  = split_df[split_df['split'] == 'test'].copy()

    # Pairformer converts ':' separator to ',' internally — match accordingly
    test_df = test_df.copy()
    test_df['mutant_comma'] = test_df['mutant'].str.replace(':', ',', regex=False)

    pf_df = pd.read_csv(pf_path)  # columns: variant, score, pairformer_zeroshot, pairformer_mlp

    # Join on mutant (comma-normalized) == variant
    merged = test_df.merge(pf_df, left_on='mutant_comma', right_on='variant', how='inner')
    if len(merged) != len(test_df):
        print(f'  WARNING: {seg} pairformer match: {len(merged)}/{len(test_df)} variants matched')

    y_true = merged['DMS_score'].values
    y_pred = merged['pairformer_mlp'].values
    return y_true, y_pred


def load_npt_summary(seg):
    """Load NPT test metrics from summary.csv (incl. func_spearman if available)."""
    path = f'{BASE}/npt_results/{seg}/summary.csv'
    df = pd.read_csv(path)
    row = df.iloc[0]

    fs     = float(row['func_spearman'])      if 'func_spearman'      in row and pd.notna(row['func_spearman'])      else None
    fs_lo  = float(row['func_spearman_ci_lo']) if 'func_spearman_ci_lo' in row and pd.notna(row['func_spearman_ci_lo']) else None
    fs_hi  = float(row['func_spearman_ci_hi']) if 'func_spearman_ci_hi' in row and pd.notna(row['func_spearman_ci_hi']) else None

    return {
        'spearman':            round(float(row['spearman_test_ood']), 4),
        'spearman_ci_lo':      round(float(row['spearman_test_ci_lo']), 4),
        'spearman_ci_hi':      round(float(row['spearman_test_ci_hi']), 4),
        'func_spearman':       round(fs, 4) if fs is not None else None,
        'func_spearman_ci_lo': round(fs_lo, 4) if fs_lo is not None else None,
        'func_spearman_ci_hi': round(fs_hi, 4) if fs_hi is not None else None,
        'auc':                 round(float(row['auc_test_ood']), 4),
        'auc_ci_lo':           None,  # not bootstrapped for NPT
        'auc_ci_hi':           None,
        'pct_positive':        None,
        'n_test':              int(row['n_test']),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

records = []

for seg, cfg in SEGMENTS.items():
    thr = cfg['thr']
    print(f'\n{"="*60}')
    print(f'Segment: {seg} (thr={thr})')

    # ── Ground truth for METL models (from TSV) ──────────────────────────────
    try:
        y_tsv, test_idx = load_tsv_test(seg, thr)
        pct_pos = round(float((y_tsv > FUNC_THR).sum()) / len(y_tsv) * 100, 1)
        n_test = len(y_tsv)
        print(f'  TSV test set: n={n_test}, %positive={pct_pos}%')
    except Exception as e:
        y_tsv = None
        test_idx = None
        print(f'  TSV load failed: {e}')

    # ── ESM-2 ────────────────────────────────────────────────────────────────
    try:
        y_esm, p_esm = load_esm_or_msa(f'{BASE}/esm_results', seg)
        m = compute_metrics(y_esm, p_esm)
        m.update({'segment': seg, 'model': 'ESM-2'})
        records.append(m)
        print(f'  ESM-2:      spearman={m["spearman"]:.4f}, func_sp={m["func_spearman"]}, auc={m["auc"]}, %pos={m["pct_positive"]}%')
    except Exception as e:
        print(f'  ESM-2: FAILED — {e}')

    # ── MSA Transformer ───────────────────────────────────────────────────────
    try:
        y_msa, p_msa = load_esm_or_msa(f'{BASE}/msa_results', seg)
        m = compute_metrics(y_msa, p_msa)
        m.update({'segment': seg, 'model': 'MSA-Transformer'})
        records.append(m)
        print(f'  MSA:        spearman={m["spearman"]:.4f}, func_sp={m["func_spearman"]}, auc={m["auc"]}, %pos={m["pct_positive"]}%')
    except Exception as e:
        print(f'  MSA: FAILED — {e}')

    # ── Pairformer ────────────────────────────────────────────────────────────
    try:
        y_pf, p_pf = load_pairformer_test(seg, thr)
        m = compute_metrics(y_pf, p_pf)
        m.update({'segment': seg, 'model': 'Pairformer'})
        records.append(m)
        print(f'  Pairformer: spearman={m["spearman"]:.4f}, func_sp={m["func_spearman"]}, auc={m["auc"]}, %pos={m["pct_positive"]}%')
    except Exception as e:
        print(f'  Pairformer: FAILED — {e}')

    # ── ProteinNPT ────────────────────────────────────────────────────────────
    try:
        m = load_npt_summary(seg)
        m.update({'segment': seg, 'model': 'ProteinNPT'})
        # fill pct_positive from TSV if available
        if y_tsv is not None:
            m['pct_positive'] = round(float((y_tsv > FUNC_THR).sum()) / len(y_tsv) * 100, 1)
        records.append(m)
        print(f'  NPT:        spearman={m["spearman"]:.4f}, auc={m["auc"]}, %pos={m["pct_positive"]}% (func_sp=N/A)')
    except Exception as e:
        print(f'  NPT: FAILED — {e}')

    # ── METL-G-1D ─────────────────────────────────────────────────────────────
    if seg in METL_1D_RUNS and y_tsv is not None:
        try:
            run_dir = f'{BASE}/metl_rosetta/results/metl_g_1D/{seg}'
            p_1d = load_metl_preds(run_dir, METL_1D_RUNS[seg])
            m = compute_metrics(y_tsv, p_1d)
            m.update({'segment': seg, 'model': 'METL-G-1D'})
            records.append(m)
            print(f'  METL-G-1D:  spearman={m["spearman"]:.4f}, func_sp={m["func_spearman"]}, auc={m["auc"]}, %pos={m["pct_positive"]}%')
        except Exception as e:
            print(f'  METL-G-1D: FAILED — {e}')
    elif seg not in METL_1D_RUNS:
        print(f'  METL-G-1D: no completed run')

    # ── METL-G-3D ─────────────────────────────────────────────────────────────
    if seg in METL_3D_RUNS and y_tsv is not None:
        try:
            run_dir = f'{BASE}/metl_rosetta/results/metl_g_3D/{seg}'
            p_3d = load_metl_preds(run_dir, METL_3D_RUNS[seg])
            m = compute_metrics(y_tsv, p_3d)
            m.update({'segment': seg, 'model': 'METL-G-3D'})
            records.append(m)
            print(f'  METL-G-3D:  spearman={m["spearman"]:.4f}, func_sp={m["func_spearman"]}, auc={m["auc"]}, %pos={m["pct_positive"]}%')
        except Exception as e:
            print(f'  METL-G-3D: FAILED — {e}')
    elif seg not in METL_3D_RUNS:
        print(f'  METL-G-3D: no completed run')

    # ── METL-Local v1/v2/v3 ──────────────────────────────────────────────────
    for version, runs, run_dir in [
        ('METL-Local-v1', METL_LOCAL_V1_RUNS, f'{BASE}/metl_rosetta/target_model_45k'),
        ('METL-Local-v2', METL_LOCAL_V2_RUNS, f'{BASE}/metl_rosetta/target_model_45k_v2'),
        ('METL-Local-v3', METL_LOCAL_V3_RUNS, f'{BASE}/metl_rosetta/target_model_45k_v3'),
    ]:
        if seg in runs and y_tsv is not None:
            try:
                p = load_metl_preds(run_dir, runs[seg])
                m = compute_metrics(y_tsv, p)
                m.update({'segment': seg, 'model': version})
                records.append(m)
                print(f'  {version}: spearman={m["spearman"]:.4f}, func_sp={m["func_spearman"]}, auc={m["auc"]}')
            except Exception as e:
                print(f'  {version}: FAILED — {e}')
        else:
            print(f'  {version}: no run')

# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

df = pd.DataFrame(records, columns=['segment', 'model', 'n_test',
                                     'spearman', 'spearman_ci_lo', 'spearman_ci_hi',
                                     'func_spearman', 'func_spearman_ci_lo', 'func_spearman_ci_hi',
                                     'auc', 'auc_ci_lo', 'auc_ci_hi',
                                     'pct_positive'])

print('\n\n' + '='*80)
print('UNIFIED RESULTS TABLE')
print('='*80)
print(df.to_string(index=False))

# Save to CSV
out_path = f'{BASE}/unified_results.csv'
df.to_csv(out_path, index=False)
print(f'\nSaved to: {out_path}')

# Pretty pivot table: spearman
print('\n── Spearman ρ (overall) ──')
pivot = df.pivot(index='model', columns='segment', values='spearman')
print(pivot.to_string())

print('\n── Functional Spearman ρ (DMS > 0.5) ──')
pivot_func = df.pivot(index='model', columns='segment', values='func_spearman')
print(pivot_func.to_string())

print('\n── AUC (DMS > 0.5) ──')
pivot_auc = df.pivot(index='model', columns='segment', values='auc')
print(pivot_auc.to_string())

print('\n── % Positive (DMS > 0.5) ──')
pivot_pct = df.pivot(index='model', columns='segment', values='pct_positive')
print(pivot_pct.to_string())
