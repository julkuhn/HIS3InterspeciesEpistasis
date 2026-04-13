#!/usr/bin/env python3
"""ProteinNPT — Non-Parametric Transformer on top of MSA Transformer embeddings.

Reference: Notin et al. 2023 "ProteinNPT: Improving Protein Property Prediction
and Design with Non-Parametric Transformers"

Pipeline:
  1. Load pre-computed MSA Transformer embeddings (X_train/val/test.npy)
     produced by msa_transformer_baseline.py.
  2. Train a Non-Parametric Transformer (NPT) that at each forward pass
     attends over a support set of (embedding, label) pairs from the training
     set to predict labels for query sequences.
  3. Evaluate with the same metrics as the MLP baselines.

NPT forward pass (one batch):
  - Support: S labeled train examples  → tokens [z_i ; y_i]  shape (S, d+1)
  - Query  : Q sequences (label masked) → tokens [z_q ; 0  ]  shape (Q, d+1)
  - Stack  : (S+Q, d+1) → project to d_npt → Transformer → read out query rows → MLP → ŷ
"""

import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt


# ============ ARGS ============
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--segment",       default="S08")
    p.add_argument("--embeddings_dir",default="/workspace/msa_results/S08",
                   help="Directory with X_train/val/test.npy and y_*.npy")
    p.add_argument("--output_dir",    default="/workspace/npt_results/S08")
    # NPT architecture
    p.add_argument("--d_npt",         type=int, default=256,
                   help="NPT hidden dimension")
    p.add_argument("--n_heads",       type=int, default=8)
    p.add_argument("--n_layers",      type=int, default=4)
    p.add_argument("--dropout",       type=float, default=0.1)
    # Training
    p.add_argument("--support_size",  type=int, default=256,
                   help="Number of labeled support examples per NPT forward pass")
    p.add_argument("--query_size",    type=int, default=64,
                   help="Number of query sequences per NPT forward pass")
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--epochs",        type=int, default=50)
    p.add_argument("--patience",      type=int, default=10)
    p.add_argument("--batch_size",    type=int, default=8,
                   help="Number of (support+query) sets per gradient step")
    p.add_argument("--n_support_sets",type=int, default=4,
                   help="At inference: how many random support subsets to average")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--auc_threshold", type=float, default=0.5)
    return p.parse_args()


args = parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Segment: {args.segment}  |  embeddings: {args.embeddings_dir}")


# ============ LOAD EMBEDDINGS ============
print("\nLade Embeddings...")
X_train = np.load(os.path.join(args.embeddings_dir, "X_train.npy"))
y_train = np.load(os.path.join(args.embeddings_dir, "y_train.npy"))
X_val   = np.load(os.path.join(args.embeddings_dir, "X_val.npy"))
y_val   = np.load(os.path.join(args.embeddings_dir, "y_val.npy"))
X_test  = np.load(os.path.join(args.embeddings_dir, "X_test.npy"))
y_test  = np.load(os.path.join(args.embeddings_dir, "y_test.npy"))

INPUT_DIM = X_train.shape[1]
print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
print(f"Embedding dim: {INPUT_DIM}")

# Convert to tensors (keep on CPU, move batches to GPU in loop)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32)


# ============ NPT MODEL ============
class NPTModel(nn.Module):
    """Non-Parametric Transformer for fitness prediction.

    Each token = [sequence_embedding ; label_or_mask]  (dim = INPUT_DIM + 1)
    Tokens are projected to d_npt, then processed by a standard Transformer
    encoder. The label of support tokens is their true DMS score (normalised);
    query tokens get a label of 0 (masked). The model reads out query
    representations and maps them to scalar predictions.
    """

    def __init__(self, input_dim, d_npt, n_heads, n_layers, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim + 1, d_npt)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_npt,
            nhead=n_heads,
            dim_feedforward=d_npt * 4,
            dropout=dropout,
            batch_first=True,   # (batch, seq, d)
            norm_first=True,    # pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_npt, d_npt // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_npt // 2, 1),
        )

    def forward(self, support_emb, support_labels, query_emb):
        """
        Args:
            support_emb    (B, S, d)  — support embeddings
            support_labels (B, S)     — support labels (normalised)
            query_emb      (B, Q, d)  — query embeddings
        Returns:
            preds (B, Q)  — predicted labels for query sequences
        """
        B, S, d = support_emb.shape
        Q = query_emb.shape[1]

        # Build tokens: [embedding ; label]
        support_tokens = torch.cat(
            [support_emb, support_labels.unsqueeze(-1)], dim=-1
        )  # (B, S, d+1)
        query_tokens = torch.cat(
            [query_emb, torch.zeros(B, Q, 1, device=query_emb.device)], dim=-1
        )  # (B, Q, d+1)  — label masked as 0

        tokens = torch.cat([support_tokens, query_tokens], dim=1)  # (B, S+Q, d+1)

        # Project and attend
        h = self.input_proj(tokens)        # (B, S+Q, d_npt)
        h = self.transformer(h)            # (B, S+Q, d_npt)

        # Read out query positions (last Q rows)
        query_h = h[:, S:, :]             # (B, Q, d_npt)
        preds = self.head(query_h).squeeze(-1)  # (B, Q)
        return preds


model = NPTModel(
    input_dim=INPUT_DIM,
    d_npt=args.d_npt,
    n_heads=args.n_heads,
    n_layers=args.n_layers,
    dropout=args.dropout,
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nNPT parameters: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
)
loss_fn = nn.MSELoss()

# Normalise labels for stable training (un-normalise for evaluation)
y_mean = float(y_train_t.mean())
y_std  = float(y_train_t.std())
y_train_norm = (y_train_t - y_mean) / y_std


# ============ TRAINING HELPERS ============
def sample_npt_batch(X_all, y_all_norm, support_size, query_size, n_sets, rng):
    """Sample n_sets independent (support, query) pairs from the training data.
    Returns tensors shaped (n_sets, support_size, d) etc."""
    n = len(X_all)
    all_idx = rng.permutation(n)

    support_embs, support_labs, query_embs, query_labs = [], [], [], []
    for i in range(n_sets):
        # Non-overlapping support and query within this set
        idx = rng.choice(n, support_size + query_size, replace=False)
        s_idx = idx[:support_size]
        q_idx = idx[support_size:]
        support_embs.append(X_all[s_idx])
        support_labs.append(y_all_norm[s_idx])
        query_embs.append(X_all[q_idx])
        query_labs.append(y_all_norm[q_idx])

    return (
        torch.stack(support_embs).to(device),  # (n_sets, S, d)
        torch.stack(support_labs).to(device),  # (n_sets, S)
        torch.stack(query_embs).to(device),    # (n_sets, Q, d)
        torch.stack(query_labs).to(device),    # (n_sets, Q)
    )


@torch.no_grad()
def predict(X_query, X_support=None, y_support_norm=None,
            support_size=None, n_sets=None, rng=None):
    """Predict DMS scores for X_query using random support subsets from train."""
    if X_support is None:
        X_support     = X_train_t
        y_support_norm = y_train_norm
    if support_size is None:
        support_size = min(args.support_size, len(X_support))
    if n_sets is None:
        n_sets = args.n_support_sets
    if rng is None:
        rng = np.random.default_rng(args.seed)

    model.eval()
    all_preds = []

    for _ in range(n_sets):
        s_idx = rng.choice(len(X_support), support_size, replace=False)
        sup_emb  = X_support[s_idx].unsqueeze(0).to(device)  # (1, S, d)
        sup_lab  = y_support_norm[s_idx].unsqueeze(0).to(device)  # (1, S)

        # Process X_query in chunks to avoid OOM
        chunk_size = 512
        preds_chunks = []
        for c_start in range(0, len(X_query), chunk_size):
            q_emb = X_query[c_start:c_start+chunk_size].unsqueeze(0).to(device)
            # Expand support to match batch dim
            s_e = sup_emb.expand(q_emb.shape[0], -1, -1)
            s_l = sup_lab.expand(q_emb.shape[0], -1)
            p = model(s_e, s_l, q_emb)  # (1, chunk)
            preds_chunks.append(p.squeeze(0).cpu())
        all_preds.append(torch.cat(preds_chunks))

    # Average over support subsets, then denormalise
    pred_norm = torch.stack(all_preds).mean(0)  # (N,)
    return (pred_norm * y_std + y_mean).numpy()


# ============ TRAINING LOOP ============
print("\nTrainiere NPT...")
rng_train = np.random.default_rng(args.seed)
best_val_sp = -np.inf
best_state  = None
patience_counter = 0

for epoch in range(1, args.epochs + 1):
    model.train()
    epoch_losses = []

    # How many gradient steps per epoch: cover ~1x the training set
    steps_per_epoch = max(1, len(X_train_t) // (args.support_size + args.query_size))
    steps_per_epoch = min(steps_per_epoch, 200)  # cap for speed

    for _ in range(steps_per_epoch):
        sup_emb, sup_lab, q_emb, q_lab = sample_npt_batch(
            X_train_t, y_train_norm,
            args.support_size, args.query_size,
            n_sets=args.batch_size,
            rng=rng_train,
        )
        optimizer.zero_grad()
        pred = model(sup_emb, sup_lab, q_emb)    # (batch_size, Q)
        loss = loss_fn(pred, q_lab)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_losses.append(loss.item())

    scheduler.step()

    # Validation every epoch
    val_pred = predict(X_val_t)
    val_sp   = spearmanr(y_val, val_pred)[0]
    train_loss = np.mean(epoch_losses)

    print(f"Epoch {epoch:3d}/{args.epochs} | loss: {train_loss:.4f} | val Spearman: {val_sp:.4f}")

    if val_sp > best_val_sp:
        best_val_sp = val_sp
        best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Restore best checkpoint
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
print(f"\nBest val Spearman: {best_val_sp:.4f}")


# ============ EVALUATION ============
def evaluate(X, y, name):
    pred  = predict(X)
    sp    = spearmanr(y, pred)[0]
    mse   = np.mean((y - pred) ** 2)
    y_bin = (y > args.auc_threshold).astype(int)
    auc   = (
        roc_auc_score(y_bin, pred)
        if y_bin.sum() > 0 and y_bin.sum() < len(y_bin)
        else float("nan")
    )
    pred_bin = (pred > args.auc_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    print(
        f"{name:20s} | Spearman: {sp:.4f} | MSE: {mse:.4f} | "
        f"AUC: {auc:.4f} | FP: {fp} / {tn+fp} neg  (FPR: {fpr:.3f})"
    )
    return sp, pred, auc


print("\n" + "=" * 62)
print(f"ProteinNPT  |  HIS3 {args.segment}  |  Phylogenetic OOD Split")
print("=" * 62)

sp_train, _,         auc_train = evaluate(X_train_t, y_train, "Train")
sp_val,   pred_val,  auc_val   = evaluate(X_val_t,   y_val,   "Validation")
sp_test,  pred_test, auc_test  = evaluate(X_test_t,  y_test,  "Test (OOD)")

print(f"\nOOD gap (train - test): {sp_train - sp_test:.4f}")


# ── Spearman for DMS > 0.5 ────────────────────────────────────────────────────
def evaluate_dms_gt0_5(y_true, y_pred, name):
    mask = y_true > 0.5
    sp = spearmanr(y_true[mask], y_pred[mask])[0]
    print(f"{name:20s} | Spearman (DMS > 0.5): {sp:.4f} | n = {mask.sum()}")


print("\n" + "=" * 62)
print("Spearman Correlation for DMS Scores > 0.5")
print("=" * 62)
evaluate_dms_gt0_5(y_train, predict(X_train_t), "Train")
evaluate_dms_gt0_5(y_val,   pred_val,           "Validation")
evaluate_dms_gt0_5(y_test,  pred_test,          "Test (OOD)")


# ============ PLOT ============
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(y_val,  pred_val,  alpha=0.3, s=5)
axes[0].plot([0, 1.5], [0, 1.5], "r--")
axes[0].set_title(f"Validation\nSpearman: {sp_val:.3f}")
axes[0].set_xlabel("True DMS Score"); axes[0].set_ylabel("Predicted")

axes[1].scatter(y_test, pred_test, alpha=0.3, s=5)
axes[1].plot([0, 1.5], [0, 1.5], "r--")
axes[1].set_title(f"Test OOD\nSpearman: {sp_test:.3f}")
axes[1].set_xlabel("True DMS Score"); axes[1].set_ylabel("Predicted")

plt.suptitle(f"ProteinNPT | HIS3 {args.segment}", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, f"npt_{args.segment}_results.png"), dpi=150)
print(f"\nPlot gespeichert!")


# ============ SAVE ============
torch.save(best_state, os.path.join(args.output_dir, "npt_model.pt"))

def spearman_boot_ci(y_true, y_pred, n_resamples=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = [
        spearmanr(y_true[idx := rng.integers(0, n, n)], y_pred[idx])[0]
        for _ in range(n_resamples)
    ]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

ci_lo, ci_hi = spearman_boot_ci(y_test, pred_test)

summary = {
    "segment":             args.segment,
    "n_train":             len(X_train),
    "n_val":               len(X_val),
    "n_test":              len(X_test),
    "spearman_train":      sp_train,
    "spearman_val":        sp_val,
    "spearman_test_ood":   sp_test,
    "spearman_test_ci_lo": ci_lo,
    "spearman_test_ci_hi": ci_hi,
    "auc_train":           auc_train,
    "auc_val":             auc_val,
    "auc_test_ood":        auc_test,
    "d_npt":               args.d_npt,
    "n_layers":            args.n_layers,
    "support_size":        args.support_size,
}
pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
print("\nFertig!")
