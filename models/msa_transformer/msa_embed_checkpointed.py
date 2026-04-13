#!/usr/bin/env python3
"""
MSA Transformer embedding with checkpointing — for large splits (e.g. ALL_combined).

Each job processes as many sequences as it can within wall time.
Progress is saved in chunks of --chunk_size sequences.
Re-running automatically resumes from the last complete chunk.

Usage (one split at a time):
  python3 msa_embed_checkpointed.py --split_name train  [other args]
  python3 msa_embed_checkpointed.py --split_name val    [other args]
  python3 msa_embed_checkpointed.py --split_name test   [other args]

When a split is fully embedded, X_{split_name}.npy / y_{split_name}.npy
are written to --output_dir.  Re-running a finished split is a no-op.
"""

import argparse, os, sys, time
import numpy as np
import pandas as pd
import torch
import esm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--segment",      default="ALL_combined")
    p.add_argument("--threshold",    default="all")
    p.add_argument("--split_dir",    default="/workspace/splits_segmentwise_species")
    p.add_argument("--output_dir",   default="/workspace/msa_results/ALL_combined")
    p.add_argument("--msa_file",     default="/workspace/data/pgen.1008079.s010.fas")
    p.add_argument("--split_name",   default="train",
                   choices=["train", "val", "test"],
                   help="Which split to embed in this job")
    p.add_argument("--chunk_size",   type=int, default=8000,
                   help="Sequences per checkpoint chunk (~1.4h on A100)")
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--n_seqs",       type=int, default=63)
    p.add_argument("--repr_layer",   type=int, default=12)
    return p.parse_args()


args        = parse_args()
SPLIT_DIR   = args.split_dir
SEGMENT     = args.segment
THRESHOLD   = args.threshold
OUTPUT_DIR  = args.output_dir
MSA_FILE    = args.msa_file
SPLIT_NAME  = args.split_name
CHUNK_SIZE  = args.chunk_size
BATCH_SIZE  = args.batch_size
N_SEQS      = args.n_seqs
REPR_LAYER  = args.repr_layer

CHUNK_DIR = os.path.join(OUTPUT_DIR, f"chunks_{SPLIT_NAME}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR,  exist_ok=True)

print(f"Segment: {SEGMENT}  Split: {SPLIT_NAME}  chunk_size: {CHUNK_SIZE}  batch: {BATCH_SIZE}")
print(f"Output:  {OUTPUT_DIR}")
print(f"Chunks:  {CHUNK_DIR}")

# ── Early exit if already finished ───────────────────────────────────────────
final_X = os.path.join(OUTPUT_DIR, f"X_{SPLIT_NAME}.npy")
final_y = os.path.join(OUTPUT_DIR, f"y_{SPLIT_NAME}.npy")
if os.path.exists(final_X) and os.path.exists(final_y):
    print(f"Already complete: {final_X} exists. Nothing to do.")
    sys.exit(0)

# ── Load split CSV ─────────────────────────────────────────────────────────────
prefixes = [
    f"super_segments_{SEGMENT}_thr{THRESHOLD}",
    f"super_segments_{SEGMENT}",
]
df = None
for pfx in prefixes:
    path = os.path.join(SPLIT_DIR, f"{pfx}_{SPLIT_NAME}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded: {path}  ({len(df):,} rows)")
        break
if df is None:
    raise FileNotFoundError(f"No CSV found for {SEGMENT}/{SPLIT_NAME} in {SPLIT_DIR}")

df["mutated_sequence"] = (
    df["mutated_sequence"].astype(str)
    .str.replace("-", "").str.replace(".", "").str.strip()
)
sequences = df["mutated_sequence"].tolist()
labels    = df["DMS_score"].values
N         = len(sequences)

# ── Find existing chunks, determine resume point ───────────────────────────────
def chunk_path(start):
    end = min(start + CHUNK_SIZE, N)
    return os.path.join(CHUNK_DIR, f"chunk_{start:07d}_{end:07d}.npy")

completed_chunks = []
start = 0
while start < N:
    cp = chunk_path(start)
    if os.path.exists(cp):
        arr = np.load(cp)
        completed_chunks.append((start, arr))
        start += CHUNK_SIZE
    else:
        break

n_done = start
print(f"Already embedded: {n_done:,} / {N:,}  ({len(completed_chunks)} chunks)")

if n_done >= N:
    print("All chunks present — merging into final arrays...")
    X_all = np.concatenate([arr for _, arr in completed_chunks], axis=0)
    np.save(final_X, X_all)
    np.save(final_y, labels)
    print(f"Saved {final_X}  shape={X_all.shape}")
    sys.exit(0)

# ── Load MSA ───────────────────────────────────────────────────────────────────
print("\nLoading MSA...")
def load_fasta(path):
    seqs, header, parts = [], None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header: seqs.append((header, "".join(parts)))
                header, parts = line[1:], []
            else:
                parts.append(line)
    if header: seqs.append((header, "".join(parts)))
    return seqs

msa_seqs = load_fasta(MSA_FILE)
print(f"MSA: {len(msa_seqs)} seqs, aln len {len(msa_seqs[0][1])}")

TEMPLATE_SPECIES = "ASHGO"
template_aligned = None
for header, aligned in msa_seqs:
    if TEMPLATE_SPECIES in header:
        template_aligned = aligned
        print(f"Template: {header[:80]}")
        break
if template_aligned is None:
    raise RuntimeError(f"{TEMPLATE_SPECIES} not found in MSA")

gap_chars = set("-. ")
ungapped_to_col = [i for i, c in enumerate(template_aligned) if c not in gap_chars]
assert len(ungapped_to_col) == 220, f"Expected 220, got {len(ungapped_to_col)}"
ALIGN_LEN = len(template_aligned)

homologs = [(h, s) for h, s in msa_seqs if TEMPLATE_SPECIES not in h][:N_SEQS]
print(f"Using {len(homologs)} homologs")

def insert_gaps(seq_220):
    aligned = ["-"] * ALIGN_LEN
    for i, col in enumerate(ungapped_to_col):
        aligned[col] = seq_220[i]
    return "".join(aligned)

# ── Load model ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
print("Loading ESM-MSA-1b...")
msa_model, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_batch_converter = msa_alphabet.get_batch_converter()
msa_model.eval().to(device)
pool_cols = torch.tensor(ungapped_to_col, dtype=torch.long).to(device)

# ── Embed remaining sequences chunk by chunk ──────────────────────────────────
print(f"\nEmbedding sequences {n_done:,} → {N:,} in chunks of {CHUNK_SIZE}...\n")

t_start = time.time()
pos = n_done

while pos < N:
    chunk_end  = min(pos + CHUNK_SIZE, N)
    cp         = chunk_path(pos)
    chunk_seqs = sequences[pos:chunk_end]
    n_chunk    = len(chunk_seqs)

    print(f"Chunk [{pos:7,} – {chunk_end:7,}]  ({n_chunk} seqs)  "
          f"elapsed={time.time()-t_start:.0f}s")
    sys.stdout.flush()

    embeddings = []
    for b in range(0, n_chunk, BATCH_SIZE):
        batch_seqs = chunk_seqs[b : b + BATCH_SIZE]
        batch_msas = [
            [("query", insert_gaps(s))] + homologs
            for s in batch_seqs
        ]
        _, _, tokens = msa_batch_converter(batch_msas)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = msa_model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
        reps = out["representations"][REPR_LAYER]          # (B, n_seqs+1, L+2, d)
        for i in range(len(batch_seqs)):
            q = reps[i, 0, 1:-1, :]                        # (512, d)
            embeddings.append(q[pool_cols].mean(0).cpu().numpy())

        if (b // BATCH_SIZE) % 50 == 0 and b > 0:
            done_global = pos + b
            elapsed     = time.time() - t_start
            rate        = done_global / elapsed if elapsed > 0 else 0
            eta_h       = (N - done_global) / rate / 3600 if rate > 0 else float("inf")
            print(f"  {done_global:7,}/{N:,}  rate={rate:.0f}/h  ETA={eta_h:.1f}h")
            sys.stdout.flush()

    # save chunk atomically (temp → rename)
    chunk_arr = np.array(embeddings)
    tmp = cp[:-4] + ".tmp.npy"
    np.save(tmp, chunk_arr)
    os.rename(tmp, cp)
    print(f"  → saved chunk {cp}  shape={chunk_arr.shape}")
    sys.stdout.flush()

    pos = chunk_end

# ── Merge all chunks into final arrays ───────────────────────────────────────
print("\nAll chunks done — merging...")
all_arrays = []
pos2 = 0
while pos2 < N:
    cp = chunk_path(pos2)
    all_arrays.append(np.load(cp))
    pos2 += CHUNK_SIZE

X_all = np.concatenate(all_arrays, axis=0)
np.save(final_X, X_all)
np.save(final_y, labels)
print(f"Saved {final_X}  shape={X_all.shape}")
print(f"Saved {final_y}  shape={labels.shape}")
print(f"Total elapsed: {(time.time()-t_start)/3600:.2f}h")
