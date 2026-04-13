#!/usr/bin/env python3
"""
MSA Pairformer: Extract embeddings + Train MLP for DMS variant effect prediction.

Step 1: Extract per-position embeddings from MSA Pairformer (frozen backbone)
Step 2: For each variant, pool mutated-position embeddings
Step 3: Train a simple MLP to predict DMS score from these embeddings

Usage:
  python3 pairformer_mlp.py \
      --msa /path/to/HIS7_YEAST.a2m \
      --dms /path/to/his3_S06.tsv \
      --split_dir /path/to/his3_S06_split \
      --weights_dir /path/to/pairformer_weights \
      --output_dir /path/to/output
"""

import argparse
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ============================================================
# 1. Argument parsing
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Pairformer MLP VEP")
    p.add_argument("--msa", required=True, help="MSA file (.a2m/.a3m/.fas)")
    p.add_argument("--dms", required=True, help="DMS TSV (variant, num_mutations, score)")
    p.add_argument("--split_dir", required=True, help="Dir with train/val/test.txt")
    p.add_argument("--weights_dir", required=True, help="Pairformer weights directory")
    p.add_argument("--output_dir", required=True, help="Output directory")
    p.add_argument("--wt", default="MTEQKALVKRITNETKIQIAISLKGGPLAIEHSIFPEKEAEAVAEQATQSQVINVHTGIGFLDHMIHALAKHSGWSLIVECIGDLHIDDHHTTEDCGIALGQAFKEALGAVRGVKRFGSGFAPLDEALSRAVVDLSNRPYAVVELGLQREKVGDLSCEMIPHFLESFAEASRITLHVDCLRGKNDHHRSESAFKALAVAIREATSPNGTNDVPSTKGVLM")
    p.add_argument("--max_msa_depth", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=256, help="MLP hidden dim")
    p.add_argument("--n_layers", type=int, default=2, help="MLP hidden layers")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--also_zero_shot", action="store_true", default=True,
                   help="Also compute zero-shot masked marginal scores")
    return p.parse_args()


# ============================================================
# 2. Extract Pairformer embeddings (frozen)
# ============================================================
def extract_embeddings(msa_path, weights_dir, max_msa_depth, device):
    """Run MSA Pairformer on the MSA and return per-position embeddings."""
    from MSA_Pairformer.model import MSAPairformer
    from MSA_Pairformer.dataset import MSA, prepare_msa_masks, aa2tok_d

    log.info("Loading MSA Pairformer model...")
    model = MSAPairformer.from_pretrained(weights_dir=weights_dir, device=device)
    model.eval()

    log.info(f"Loading MSA from {msa_path}...")
    np.random.seed(42)

    # Try loading with hhfilter first, fall back to random subsampling
    try:
        msa_obj = MSA(
            msa_file_path=msa_path,
            max_seqs=max_msa_depth,
            max_length=10240,
            max_tokens=np.inf,
            diverse_select_method="hhfilter",
            hhfilter_kwargs={"binary": "hhfilter"}
        )
    except Exception as e:
        log.warning(f"hhfilter failed ({e}), using random subsampling")
        msa_obj = MSA(
            msa_file_path=msa_path,
            max_seqs=max_msa_depth,
            max_length=10240,
            max_tokens=np.inf,
            diverse_select_method="greedy",
        )

    msa_tokenized_t = msa_obj.diverse_tokenized_msa
    n_seqs, seq_len = msa_tokenized_t.shape
    log.info(f"MSA: {n_seqs} sequences, length {seq_len}")

    msa_onehot = torch.nn.functional.one_hot(
        msa_tokenized_t, num_classes=len(aa2tok_d)
    ).unsqueeze(0).float().to(device)

    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(
        msa_tokenized_t.unsqueeze(0)
    )
    mask = mask.to(device)
    msa_mask = msa_mask.to(device)
    full_mask = full_mask.to(device)
    pairwise_mask = pairwise_mask.to(device)

    log.info("Running forward pass...")
    with torch.no_grad():
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda" if "cuda" in str(device) else "cpu"):
            res = model(
                msa=msa_onehot.to(torch.bfloat16),
                mask=mask,
                msa_mask=msa_mask,
                full_mask=full_mask,
                pairwise_mask=pairwise_mask,
                return_contacts=False,
            )

    # Query sequence embedding: (L, 464)
    seq_embedding = res["final_msa_repr"][0, 0, :, :].float().cpu()
    # Logits for zero-shot: (L, 26)
    logits = res["logits"][0, 0, :, :].float().cpu()

    log.info(f"Embedding shape: {seq_embedding.shape}")
    log.info(f"Logits shape: {logits.shape}")

    return seq_embedding, logits, aa2tok_d


# ============================================================
# 3. Build variant features from embeddings
# ============================================================
def parse_variant(variant_str, wt_seq):
    """Parse 'A5G,L10P' -> list of (pos_0based, wt_aa, mut_aa)"""
    mutations = []
    for mut in variant_str.split(","):
        mut = mut.strip()
        if not mut:
            continue
        wt_aa = mut[0]
        mut_aa = mut[-1]
        pos = int(mut[1:-1]) - 1  # 0-based
        if pos < len(wt_seq) and wt_seq[pos] == wt_aa:
            mutations.append((pos, wt_aa, mut_aa))
        else:
            log.warning(f"Skipping mutation {mut}: pos {pos+1} wt mismatch")
    return mutations


def build_variant_features(variants, wt_seq, seq_embedding):
    """
    For each variant, create a feature vector by:
    - Mean-pooling embeddings at mutated positions
    - Concatenating with count of mutations
    """
    emb_dim = seq_embedding.shape[1]
    features = []

    for variant_str in variants:
        mutations = parse_variant(variant_str, wt_seq)
        if not mutations:
            # WT or unparseable - use mean of all positions
            feat = seq_embedding.mean(dim=0)
        elif len(mutations) == 1:
            pos = mutations[0][0]
            feat = seq_embedding[pos]
        else:
            positions = [m[0] for m in mutations]
            feat = seq_embedding[positions].mean(dim=0)

        # Append mutation count as extra feature
        n_mut = len(mutations)
        feat = torch.cat([feat, torch.tensor([float(n_mut)])])
        features.append(feat)

    return torch.stack(features)


def compute_zero_shot_scores(variants, wt_seq, logits, aa2tok):
    """Compute masked marginal zero-shot scores from logits."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).numpy()
    scores = []

    for variant_str in variants:
        mutations = parse_variant(variant_str, wt_seq)
        if not mutations:
            scores.append(0.0)
            continue
        score = 0.0
        for pos, wt_aa, mut_aa in mutations:
            wt_aa_upper = wt_aa.upper()
            mut_aa_upper = mut_aa.upper()
            if wt_aa_upper in aa2tok and mut_aa_upper in aa2tok:
                wt_idx = aa2tok[wt_aa_upper]
                mut_idx = aa2tok[mut_aa_upper]
                score += log_probs[pos, mut_idx] - log_probs[pos, wt_idx]
        scores.append(score)

    return np.array(scores)


# ============================================================
# 4. Dataset & MLP
# ============================================================
class VariantDataset(Dataset):
    def __init__(self, features, scores):
        self.features = features
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return self.features[idx], self.scores[idx]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_layers=2, dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# 5. Training loop
# ============================================================
def train_mlp(model, train_loader, val_loader, optimizer, scheduler, epochs, patience, device):
    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "val_rho": []}

    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_preds, val_true = [], []
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_losses.append(loss.item())
                val_preds.extend(pred.cpu().numpy())
                val_true.extend(y.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_rho = spearmanr(val_true, val_preds).correlation if len(val_true) > 5 else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rho"].append(val_rho)

        if scheduler:
            scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            log.info(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_rho={val_rho:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return history


# ============================================================
# 6. Main
# ============================================================
def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Extract embeddings ----
    seq_embedding, logits, aa2tok = extract_embeddings(
        args.msa, args.weights_dir, args.max_msa_depth, device
    )

    # ---- Load DMS data ----
    log.info(f"Loading DMS from {args.dms}")
    dms = pd.read_csv(args.dms, sep="\t")
    log.info(f"Total variants: {len(dms)}")

    # Load splits
    splits = {}
    for name in ["train", "val", "test"]:
        fn = os.path.join(args.split_dir, f"{name}.txt")
        if os.path.exists(fn):
            with open(fn) as f:
                splits[name] = [int(x.strip()) for x in f.readlines()]
            log.info(f"  {name}: {len(splits[name])}")

    # ---- Zero-shot scores ----
    if args.also_zero_shot:
        log.info("Computing zero-shot scores...")
        zs_scores = compute_zero_shot_scores(dms["variant"].tolist(), args.wt, logits, aa2tok)
        dms["pairformer_zeroshot"] = zs_scores

        for name, idxs in splits.items():
            sub = dms.iloc[idxs]
            rho = spearmanr(sub["score"], sub["pairformer_zeroshot"]).correlation
            log.info(f"  Zero-shot {name}: Spearman rho = {rho:.4f} (n={len(sub)})")

        rho_all = spearmanr(dms["score"], dms["pairformer_zeroshot"]).correlation
        log.info(f"  Zero-shot overall: Spearman rho = {rho_all:.4f}")

    # ---- Build features ----
    log.info("Building variant features...")
    features = build_variant_features(dms["variant"].tolist(), args.wt, seq_embedding)
    scores_tensor = torch.tensor(dms["score"].values, dtype=torch.float32)
    input_dim = features.shape[1]
    log.info(f"Feature dim: {input_dim} (464 emb + 1 n_mut)")

    # ---- Prepare data loaders ----
    train_idx = splits.get("train", [])
    val_idx = splits.get("val", [])
    test_idx = splits.get("test", [])

    train_ds = VariantDataset(features[train_idx], scores_tensor[train_idx])
    val_ds = VariantDataset(features[val_idx], scores_tensor[val_idx])
    test_ds = VariantDataset(features[test_idx], scores_tensor[test_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ---- Train MLP ----
    log.info(f"Training MLP: {args.n_layers} layers, hidden={args.hidden_dim}, dropout={args.dropout}")
    mlp = MLP(input_dim, args.hidden_dim, args.n_layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = train_mlp(mlp, train_loader, val_loader, optimizer, scheduler,
                        args.epochs, args.patience, device)

    # ---- Evaluate ----
    mlp.eval()
    results = {}
    for split_name, loader, idxs in [("train", train_loader, train_idx),
                                      ("val", val_loader, val_idx),
                                      ("test", test_loader, test_idx)]:
        all_preds, all_true = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                pred = mlp(x)
                all_preds.extend(pred.cpu().numpy())
                all_true.extend(y.numpy())

        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        
        rho = spearmanr(all_true, all_preds).correlation
        
        # Filter for score > 0.5
        mask_high = all_true > 0.5
        if mask_high.sum() > 1:
            rho_high = spearmanr(all_true[mask_high], all_preds[mask_high]).correlation
            n_high = mask_high.sum()
        else:
            rho_high = np.nan
            n_high = mask_high.sum()
        
        results[split_name] = {
            "spearman": rho,
            "n": len(all_true),
            "spearman_score_gt_05": rho_high,
            "n_score_gt_05": int(n_high)
        }
        log.info(f"  MLP {split_name}: Spearman rho = {rho:.4f} (n={len(all_true)})")
        log.info(f"                   Spearman rho (score>0.5) = {rho_high:.4f} (n={n_high})")

        # Store predictions
        dms.loc[dms.index[idxs], "pairformer_mlp"] = all_preds

    # ---- Save everything ----
    # Predictions
    out_csv = os.path.join(args.output_dir, "pairformer_predictions.csv")
    dms.to_csv(out_csv, index=False)
    log.info(f"Saved predictions: {out_csv}")

    # Model
    torch.save(mlp.state_dict(), os.path.join(args.output_dir, "mlp_model.pt"))

    # Embeddings (for reuse)
    torch.save(seq_embedding, os.path.join(args.output_dir, "seq_embedding.pt"))

    # Results summary
    summary = {
        "results": results,
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\n=== SUMMARY ===")
    for split_name, r in results.items():
        log.info(f"  {split_name}:")
        log.info(f"    Spearman (all): {r['spearman']:.4f} (n={r['n']})")
        log.info(f"    Spearman (score>0.5): {r['spearman_score_gt_05']:.4f} (n={r['n_score_gt_05']})")
    log.info(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()