#!/usr/bin/env python3
"""MSA Transformer (100M) Baseline für HIS3 DMS (phylogenetic OOD split).

Approach:
  1. Load pre-computed MSA (pgen.1008079.s010.fas, 392 homologs, 512 alignment cols).
  2. Use Ashbya gossypii (closest 220aa homolog) gap pattern as template to
     insert gaps into each DMS variant sequence.
  3. For each variant: prepend gapped sequence as row 0 of the MSA, feed to
     ESM-MSA-1b, take mean-pool of the first-row token embeddings.
  4. Train sklearn MLP on these embeddings (identical setup to ESM-2 baseline).
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import esm
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ============ ARGS ============
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--segment",       default="S06")
    p.add_argument("--threshold",     default="0.18")
    p.add_argument("--split_dir",     default="/workspace/splits_segmentwise_species")
    p.add_argument("--output_dir",    default="/workspace/msa_results/S06")
    p.add_argument("--msa_file",      default="/workspace/data/pgen.1008079.s010.fas")
    p.add_argument("--use_fragment",  action="store_true", default=False)
    p.add_argument("--frag_start",    type=int, default=0,   help="0-indexed, inclusive")
    p.add_argument("--frag_end",      type=int, default=220, help="0-indexed, exclusive")
    p.add_argument("--n_seqs",        type=int, default=63,
                   help="Number of homologs to include (query + n_seqs <= 64 for memory)")
    p.add_argument("--batch_size",    type=int, default=2,
                   help="Number of MSAs per GPU batch (keep low, each MSA is large)")
    p.add_argument("--auc_threshold", type=float, default=0.5)
    p.add_argument("--repr_layer",    type=int, default=12,
                   help="MSA Transformer has 12 layers; use last by default")
    return p.parse_args()


args = parse_args()
SPLIT_DIR     = args.split_dir
SEGMENT       = args.segment
THRESHOLD     = args.threshold
OUTPUT_DIR    = args.output_dir
MSA_FILE      = args.msa_file
USE_FRAGMENT  = args.use_fragment
FRAG_START    = args.frag_start
FRAG_END      = args.frag_end
N_SEQS        = args.n_seqs
BATCH_SIZE    = args.batch_size
AUC_THRESHOLD = args.auc_threshold
REPR_LAYER    = args.repr_layer

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Segment: {SEGMENT}, Threshold: {THRESHOLD}, Fragment: {USE_FRAGMENT} [{FRAG_START}:{FRAG_END}]")
print(f"MSA homologs: {N_SEQS}, batch_size: {BATCH_SIZE}")


# ============ DATA LOADING ============
prefixes = [
    f"super_segments_{SEGMENT}_thr{THRESHOLD}",
    f"super_segments_{SEGMENT}",
]

train_df = val_df = test_df = None
for prefix in prefixes:
    train_path = os.path.join(SPLIT_DIR, f"{prefix}_train.csv")
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        val_df   = pd.read_csv(os.path.join(SPLIT_DIR, f"{prefix}_val.csv"))
        test_df  = pd.read_csv(os.path.join(SPLIT_DIR, f"{prefix}_test.csv"))
        print(f"Loaded: {prefix}_*.csv")
        break

if train_df is None:
    raise FileNotFoundError(f"No split files found in {SPLIT_DIR} for {SEGMENT}")

for df in [train_df, val_df, test_df]:
    df["mutated_sequence"] = (
        df["mutated_sequence"].astype(str)
        .str.replace("-", "").str.replace(".", "").str.strip()
    )

print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test (OOD): {len(test_df):,}")


# ============ MSA LOADING ============
def load_msa_fasta(path):
    """Returns list of (header, sequence) tuples."""
    seqs = []
    header = None
    seq_parts = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    seqs.append((header, "".join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        seqs.append((header, "".join(seq_parts)))
    return seqs


print("\nLade MSA...")
msa_seqs = load_msa_fasta(MSA_FILE)
print(f"MSA: {len(msa_seqs)} Sequenzen, Alignment-Länge: {len(msa_seqs[0][1])}")

# ── Find Ashbya gossypii as gap-pattern template ───────────────────────────────
# It is the closest 220aa homolog to S. cerevisiae HIS3 in this MSA.
TEMPLATE_SPECIES = "ASHGO"
template_aligned = None
for header, aligned in msa_seqs:
    if TEMPLATE_SPECIES in header:
        template_aligned = aligned
        print(f"Template: {header[:80]}")
        ungapped = aligned.replace("-", "").replace(".", "").replace("X", "")
        print(f"  ungapped length: {len(ungapped)}")
        break

if template_aligned is None:
    raise RuntimeError(f"Template sequence ({TEMPLATE_SPECIES}) not found in MSA")

# Derive gap-insertion pattern: for each position in the ungapped seq,
# store the corresponding column index in the alignment.
gap_chars = set("-. ")
ungapped_to_col = []  # ungapped_to_col[i] = alignment column for residue i
for col_idx, char in enumerate(template_aligned):
    if char not in gap_chars:
        ungapped_to_col.append(col_idx)

assert len(ungapped_to_col) == 220, (
    f"Expected 220 non-gap positions in template, got {len(ungapped_to_col)}"
)
ALIGN_LEN = len(template_aligned)
print(f"Alignment length: {ALIGN_LEN}, template non-gap cols: {len(ungapped_to_col)}")


def insert_gaps(seq_220):
    """Map a 220aa sequence onto the full alignment (512 cols) using the
    Ashbya gossypii column pattern. Positions not covered get '-'."""
    if len(seq_220) != 220:
        raise ValueError(f"Expected 220aa, got {len(seq_220)}")
    aligned = ["-"] * ALIGN_LEN
    for res_idx, col_idx in enumerate(ungapped_to_col):
        aligned[col_idx] = seq_220[res_idx]
    return "".join(aligned)


# Pre-build the homolog pool (raw aligned strings, no modification needed)
# Subsample to N_SEQS homologs (skip template itself to avoid duplication)
homolog_pool = [
    (h, s) for h, s in msa_seqs if TEMPLATE_SPECIES not in h
]
# Use first N_SEQS as fixed set (or all if fewer available)
homologs = homolog_pool[: N_SEQS]
print(f"Homolog pool: {len(homolog_pool)}, using: {len(homologs)}")


# ============ MSA TRANSFORMER LOADING ============
print("\nLade ESM-MSA-1b...")
msa_model, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_batch_converter = msa_alphabet.get_batch_converter()
msa_model.eval()
msa_model = msa_model.to(device)
print("Modell geladen!")


# ============ EMBEDDING FUNCTION ============
# Determine which residue indices to mean-pool over.
# Fragment mode: only pool over positions FRAG_START..FRAG_END of the full seq.
# Full mode: pool over all 220 positions.
# Either way we always feed the full 220aa (gap-inserted) sequence to the model.
if USE_FRAGMENT:
    pool_cols = torch.tensor(
        ungapped_to_col[FRAG_START:FRAG_END], dtype=torch.long
    )
    print(f"\nUsing fragment: positions {FRAG_START+1}-{FRAG_END} "
          f"(length {FRAG_END - FRAG_START})")
else:
    pool_cols = torch.tensor(ungapped_to_col, dtype=torch.long)
    print("\nUsing full sequence (220 aa)")


def get_msa_embeddings(sequences, batch_size=BATCH_SIZE):
    """Compute MSA Transformer embeddings for a list of 220aa sequences.

    For each sequence:
      - Insert gaps to create the aligned query (512 cols).
      - Prepend as row 0 of the fixed homolog MSA.
      - Run MSA Transformer.
      - Mean-pool the first-row token embeddings over pool_cols.
    """
    embeddings = []
    n = len(sequences)
    _pool_cols = pool_cols.to(device)

    for batch_start in range(0, n, batch_size):
        batch_seqs = sequences[batch_start : batch_start + batch_size]

        batch_msas = []
        for query_seq in batch_seqs:
            aligned_query = insert_gaps(query_seq)
            msa_input = [("query", aligned_query)] + homologs
            batch_msas.append(msa_input)

        _, _, batch_tokens = msa_batch_converter(batch_msas)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = msa_model(
                batch_tokens,
                repr_layers=[REPR_LAYER],
                return_contacts=False,
            )

        # (batch, n_seqs+1, seq_len+2, d_model) — skip BOS/EOS
        token_reps = results["representations"][REPR_LAYER]

        for b_idx in range(len(batch_seqs)):
            query_reps = token_reps[b_idx, 0, 1:-1, :]   # (512, d_model)
            residue_reps = query_reps[_pool_cols]          # (n_pool, d_model)
            emb = residue_reps.mean(dim=0).cpu().numpy()
            embeddings.append(emb)

        if batch_start % 500 == 0 and batch_start > 0:
            print(f"  {batch_start}/{n}")

    return np.array(embeddings)


# ============ COMPUTE EMBEDDINGS ============
print("\nBerechne Embeddings...")

print("Train...")
X_train = get_msa_embeddings(train_df["mutated_sequence"].tolist())
y_train = train_df["DMS_score"].values

print("Val...")
X_val = get_msa_embeddings(val_df["mutated_sequence"].tolist())
y_val = val_df["DMS_score"].values

print("Test OOD...")
X_test = get_msa_embeddings(test_df["mutated_sequence"].tolist())
y_test = test_df["DMS_score"].values

for name, arr in [("X_train", X_train), ("y_train", y_train),
                   ("X_val", X_val),   ("y_val", y_val),
                   ("X_test", X_test), ("y_test", y_test)]:
    np.save(os.path.join(OUTPUT_DIR, f"{name}.npy"), arr)
print("Embeddings gespeichert!")


# ============ MLP TRAINING ============
print("\nTrainiere MLP...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

mlp = MLPRegressor(
    hidden_layer_sizes=(512, 256, 128),
    activation="relu",
    solver="adam",
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=True,
)
mlp.fit(X_train_scaled, y_train)


# ============ EVALUATION ============
def evaluate(X, y, name):
    pred = mlp.predict(X)
    sp   = spearmanr(y, pred)[0]
    mse  = np.mean((y - pred) ** 2)
    y_bin    = (y > AUC_THRESHOLD).astype(int)
    pred_bin = (pred > AUC_THRESHOLD).astype(int)
    auc = (
        roc_auc_score(y_bin, pred)
        if y_bin.sum() > 0 and y_bin.sum() < len(y_bin)
        else float("nan")
    )
    tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    print(
        f"{name:20s} | Spearman: {sp:.4f} | MSE: {mse:.4f} | "
        f"AUC: {auc:.4f} | FP: {fp} / {tn+fp} neg  (FPR: {fpr:.3f})"
    )
    return sp, pred, auc


print("\n" + "=" * 60)
print(f"MSA Transformer 100M + MLP  |  HIS3 {SEGMENT}  |  Phylogenetic OOD Split")
print("=" * 60)

sp_train, _,         auc_train = evaluate(X_train_scaled, y_train, "Train")
sp_val,   pred_val,  auc_val   = evaluate(X_val_scaled,   y_val,   "Validation")
sp_test,  pred_test, auc_test  = evaluate(X_test_scaled,  y_test,  "Test (OOD)")

print(f"\nOOD gap (train - test): {sp_train - sp_test:.4f}")


# ── Spearman for DMS > 0.5 ────────────────────────────────────────────────────
def evaluate_dms_gt0_5(y_true, y_pred, name):
    mask = y_true > 0.5
    sp = spearmanr(y_true[mask], y_pred[mask])[0]
    print(f"{name:20s} | Spearman (DMS > 0.5): {sp:.4f} | n = {mask.sum()}")


print("\n" + "=" * 60)
print("Spearman Correlation for DMS Scores > 0.5")
print("=" * 60)
evaluate_dms_gt0_5(y_train, mlp.predict(X_train_scaled), "Train")
evaluate_dms_gt0_5(y_val,   pred_val,                    "Validation")
evaluate_dms_gt0_5(y_test,  pred_test,                   "Test (OOD)")


# ============ PLOT ============
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_val, pred_val, alpha=0.3, s=5)
axes[0].plot([0, 1.5], [0, 1.5], "r--")
axes[0].set_title(f"Validation\nSpearman: {sp_val:.3f}")
axes[0].set_xlabel("True DMS Score")
axes[0].set_ylabel("Predicted")

axes[1].scatter(y_test, pred_test, alpha=0.3, s=5)
axes[1].plot([0, 1.5], [0, 1.5], "r--")
axes[1].set_title(f"Test OOD\nSpearman: {sp_test:.3f}")
axes[1].set_xlabel("True DMS Score")
axes[1].set_ylabel("Predicted")

plt.suptitle(
    f"MSA Transformer 100M + MLP | HIS3 {SEGMENT} | Fragment={USE_FRAGMENT}",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"msa_transformer_{SEGMENT}_results.png"), dpi=150)
print(f"\nPlot gespeichert!")

# ── Save model ────────────────────────────────────────────────────────────────
with open(os.path.join(OUTPUT_DIR, "mlp_model.pkl"), "wb") as f:
    pickle.dump(mlp, f)
with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)


# ── Bootstrap 95% CI für Spearman auf Test ───────────────────────────────────
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
    "segment":            SEGMENT,
    "threshold":          THRESHOLD,
    "use_fragment":       USE_FRAGMENT,
    "n_train":            len(train_df),
    "n_val":              len(val_df),
    "n_test":             len(test_df),
    "spearman_train":     sp_train,
    "spearman_val":       sp_val,
    "spearman_test_ood":  sp_test,
    "spearman_test_ci_lo": ci_lo,
    "spearman_test_ci_hi": ci_hi,
    "auc_train":          auc_train,
    "auc_val":            auc_val,
    "auc_test_ood":       auc_test,
    "n_homologs":         N_SEQS,
}
pd.DataFrame([summary]).to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

print("\nFertig!")
