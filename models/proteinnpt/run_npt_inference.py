#!/usr/bin/env python3
"""
Run inference with saved NPT models to extract test predictions.
Saves test_predictions.npy in each npt_results/{seg}/ directory,
then updates summary.csv with func_spearman and func_spearman_ci_lo/hi.
"""

import sys
sys.path.insert(0, '/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra/.pip_packages')

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

BASE = '/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra'

SEGMENTS = ['S02', 'S03', 'S04', 'S05', 'S05_swap', 'S06', 'S07', 'S08', 'S12', 'S12_swap']

N_BOOT        = 1000
N_SUPPORT_SETS = 4
SUPPORT_SIZE  = 256
SEED          = 42
FUNC_THR      = 0.5

device = torch.device('cpu')  # CPU inference only outside container


class NPTModel(nn.Module):
    def __init__(self, input_dim, d_npt, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim + 1, d_npt)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_npt, nhead=n_heads,
            dim_feedforward=d_npt * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_npt, d_npt // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_npt // 2, 1),
        )

    def forward(self, support_emb, support_labels, query_emb):
        B, S, d = support_emb.shape
        Q = query_emb.shape[1]
        support_tokens = torch.cat([support_emb, support_labels.unsqueeze(-1)], dim=-1)
        query_tokens   = torch.cat([query_emb,   torch.zeros(B, Q, 1)], dim=-1)
        tokens = torch.cat([support_tokens, query_tokens], dim=1)
        h = self.input_proj(tokens)
        h = self.transformer(h)
        return self.head(h[:, S:, :]).squeeze(-1)


@torch.no_grad()
def predict(model, X_query_t, X_support_t, y_support_norm,
            support_size=SUPPORT_SIZE, n_sets=N_SUPPORT_SETS, seed=SEED):
    model.eval()
    rng = np.random.default_rng(seed)
    n = len(X_support_t)
    sup_size = min(support_size, n)
    all_preds = []
    for _ in range(n_sets):
        idx = rng.choice(n, sup_size, replace=False)
        sup_emb = X_support_t[idx].unsqueeze(0)   # (1, S, d)
        sup_lab = y_support_norm[idx].unsqueeze(0) # (1, S)

        # Process query in chunks to avoid OOM
        chunk, preds = 512, []
        for i in range(0, len(X_query_t), chunk):
            q = X_query_t[i:i+chunk].unsqueeze(0)  # (1, Q, d)
            preds.append(model(sup_emb, sup_lab, q).squeeze(0))
        all_preds.append(torch.cat(preds))

    pred_norm = torch.stack(all_preds).mean(0).numpy()
    return pred_norm


def bootstrap_ci_spearman(y_true, y_pred, n=N_BOOT, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    stats = [spearmanr(y_true[rng.choice(idx, len(idx), replace=True)],
                       y_pred[rng.choice(idx, len(idx), replace=True)])[0]
             for _ in range(n)]
    stats = [spearmanr(y_true[s := rng.choice(idx, len(idx), replace=True)],
                       y_pred[s])[0] for _ in range(n)]
    return (round(float(np.percentile(stats, 100*alpha/2)), 4),
            round(float(np.percentile(stats, 100*(1-alpha/2))), 4))


def bootstrap_ci_func_spearman(y_true, y_pred, thr=FUNC_THR, n=N_BOOT, alpha=0.05, seed=SEED):
    mask = y_true > thr
    if mask.sum() < 10:
        return None, None
    yt, yp = y_true[mask], y_pred[mask]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(yt))
    stats = [spearmanr(yt[s := rng.choice(idx, len(idx), replace=True)],
                       yp[s])[0] for _ in range(n)]
    return (round(float(np.percentile(stats, 100*alpha/2)), 4),
            round(float(np.percentile(stats, 100*(1-alpha/2))), 4))


for seg in SEGMENTS:
    npt_dir = f'{BASE}/npt_results/{seg}'
    msa_dir = f'{BASE}/msa_results/{seg}'
    model_path = f'{npt_dir}/npt_model.pt'

    if not os.path.exists(model_path):
        print(f'{seg}: no model file, skipping')
        continue

    print(f'\n{seg}: loading model...')
    summary = pd.read_csv(f'{npt_dir}/summary.csv').iloc[0]
    d_npt   = int(summary['d_npt'])
    n_layers = int(summary['n_layers'])
    INPUT_DIM = np.load(f'{msa_dir}/X_train.npy').shape[1]

    n_heads = max(h for h in [8, 4, 2, 1] if d_npt % h == 0)

    model = NPTModel(INPUT_DIM, d_npt, n_heads, n_layers).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    X_train = torch.tensor(np.load(f'{msa_dir}/X_train.npy'), dtype=torch.float32)
    y_train = torch.tensor(np.load(f'{msa_dir}/y_train.npy'), dtype=torch.float32)
    X_test  = torch.tensor(np.load(f'{msa_dir}/X_test.npy'),  dtype=torch.float32)
    y_test  = np.load(f'{msa_dir}/y_test.npy')

    y_mean = float(y_train.mean())
    y_std  = float(y_train.std())
    y_train_norm = (y_train - y_mean) / y_std

    # Predict (normalised), then un-normalise
    pred_norm = predict(model, X_test, X_train, y_train_norm)
    pred = pred_norm * y_std + y_mean

    # Save predictions
    np.save(f'{npt_dir}/test_predictions.npy', pred)

    # Compute functional Spearman
    mask = y_test > FUNC_THR
    if mask.sum() >= 10:
        fs, _ = spearmanr(y_test[mask], pred[mask])
        fs_lo, fs_hi = bootstrap_ci_func_spearman(y_test, pred)
    else:
        fs = float('nan')
        fs_lo, fs_hi = None, None

    overall_sp, _ = spearmanr(y_test, pred)
    print(f'  overall spearman:  {overall_sp:.4f}  (saved: {float(summary["spearman_test_ood"]):.4f})')
    print(f'  func_spearman:     {fs:.4f}  (n_pos={mask.sum()})')
    print(f'  func_spearman CI:  [{fs_lo}, {fs_hi}]')

    # Update summary.csv
    df_s = pd.read_csv(f'{npt_dir}/summary.csv')
    df_s['func_spearman']      = round(float(fs), 4) if not np.isnan(fs) else None
    df_s['func_spearman_ci_lo'] = fs_lo
    df_s['func_spearman_ci_hi'] = fs_hi
    df_s.to_csv(f'{npt_dir}/summary.csv', index=False)
    print(f'  Updated {npt_dir}/summary.csv')

print('\nDone.')
