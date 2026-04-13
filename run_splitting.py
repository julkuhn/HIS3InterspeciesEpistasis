#!/usr/bin/env python3
"""
Phylogenetic species-cluster split for HIS3 DMS data.

Erzeugt:
  - Pro Segment den besten Train/Val/Test-Split (Sweep über Clustering-Thresholds)
  - Einen kombinierten Split über alle Segmente
  - sweep_results.csv mit allen Sweep-Metriken

Ausführung lokal oder als SLURM-Job (siehe run_splitting.sh).
"""

import os
import re
import ast
import argparse
import logging
import numpy as np
import pandas as pd

from Bio import SeqIO
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import wasserstein_distance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================
# Config / Defaults
# ============================================================
BASE = "/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra"

DEFAULTS = dict(
    csv_path   = os.path.join(BASE, "data/HIS7_YEAST_Pokusaeva_2019_with_segments.csv"),
    fasta_path = os.path.join(BASE, "data/pgen.1008079.s010.fas"),
    out_dir    = os.path.join(BASE, "splits_segmentwise_species"),
    wt_seq     = ("MTEQKALVKRITNETKIQIAISLKGGPLAIEHSIFPEKEAEAVAEQATQSQVINVHTGIGFLDHMIHALA"
                  "KHSGWSLIVECIGDLHIDDHHTTEDCGIALGQAFKEALGAVRGVKRFGSGFAPLDEALSRAVVDLSNRPY"
                  "AVVELGLQREKVGDLSCEMIPHFLESFAEASRITLHVDCLRGKNDHHRSESAFKALAVAIREATSPNGTND"
                  "VPSTKGVLM"),
    random_seed              = 67,
    test_fraction            = 0.20,
    val_fraction_within_train= 0.10,
    chunk_size               = 2048,
    min_train                = 5000,
    min_test                 = 2000,
    max_test_fraction        = 0.50,
    max_val_fraction         = 0.30,
    min_ood_gap              = 0.05,
    func_threshold           = 0.50,
    segment_level            = "super_segments",
    seq_col                  = "mutated_sequence",
    score_col                = "DMS_score",
    mutpos_col               = "mut_pos",
    segment_type_col         = "segment_type",
    allowed_types            = "within_segment,single_subsegment",
    plot                     = False,
)

THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35]


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv_path",    default=DEFAULTS["csv_path"])
    p.add_argument("--fasta_path",  default=DEFAULTS["fasta_path"])
    p.add_argument("--out_dir",     default=DEFAULTS["out_dir"])
    p.add_argument("--wt_seq",      default=DEFAULTS["wt_seq"])
    p.add_argument("--random_seed", type=int,   default=DEFAULTS["random_seed"])
    p.add_argument("--test_fraction",            type=float, default=DEFAULTS["test_fraction"])
    p.add_argument("--val_fraction_within_train",type=float, default=DEFAULTS["val_fraction_within_train"])
    p.add_argument("--chunk_size",  type=int,   default=DEFAULTS["chunk_size"])
    p.add_argument("--min_train",   type=int,   default=DEFAULTS["min_train"])
    p.add_argument("--min_test",    type=int,   default=DEFAULTS["min_test"])
    p.add_argument("--max_test_fraction", type=float, default=DEFAULTS["max_test_fraction"])
    p.add_argument("--max_val_fraction",  type=float, default=DEFAULTS["max_val_fraction"])
    p.add_argument("--min_ood_gap",       type=float, default=DEFAULTS["min_ood_gap"])
    p.add_argument("--func_threshold",    type=float, default=DEFAULTS["func_threshold"])
    p.add_argument("--segment_level",  default=DEFAULTS["segment_level"])
    p.add_argument("--plot", action="store_true", default=DEFAULTS["plot"],
                   help="Erzeuge Diagnose-Plots (braucht Matplotlib/Seaborn)")
    p.add_argument("--no_relaxed_fallback", action="store_true", default=False,
                   help="Deaktiviere Relaxed-Fallback für Segmente ohne strikten Split")
    return p.parse_args()


# ============================================================
# 1. Daten laden & parsen
# ============================================================
def load_pg(csv_path, segment_level, seq_col, mutpos_col, segment_type_col, allowed_types):
    pg = pd.read_csv(csv_path)
    log.info(f"CSV geladen: {len(pg)} Zeilen")

    def ensure_list(x):
        if isinstance(x, list): return x
        if pd.isna(x): return []
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("[") and s.endswith("]"):
                try: return ast.literal_eval(s)
                except: pass
            if ";" in s: return [t for t in s.split(";") if t]
            if "," in s: return [t.strip() for t in s.split(",") if t.strip()]
        return [x]

    def to_int_list(lst):
        out = []
        for v in lst:
            try: out.append(int(v))
            except: out.extend([int(t) for t in re.findall(r"\d+", str(v))])
        return out

    pg[mutpos_col]    = pg[mutpos_col].apply(ensure_list).apply(to_int_list)
    pg[segment_level] = pg[segment_level].apply(ensure_list)
    pg[seq_col]       = pg[seq_col].astype(str).str.replace("-","",regex=False).str.replace(".","",regex=False).str.strip()

    before = len(pg)
    pg = pg.loc[pg[segment_type_col].isin(set(allowed_types.split(",")))].copy()
    log.info(f"Filtered {before} -> {len(pg)} Zeilen (ALLOWED_TYPES)")

    def pick_segment_id(lst):
        if not lst: return np.nan
        if len(lst) == 1: return str(lst[0])
        return ";".join([str(x) for x in lst])

    pg["_segment_id"] = pg[segment_level].apply(pick_segment_id)
    pg = pg.dropna(subset=["_segment_id", seq_col]).copy()
    log.info(f"Segmente: {pg['_segment_id'].nunique()}")
    return pg


# ============================================================
# 2. MSA laden + Referenz finden
# ============================================================
def load_msa(fasta_path, wt_seq):
    rows = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        rows.append({"id": rec.id, "header": rec.description, "seq_aln": str(rec.seq)})
    msa = pd.DataFrame(rows)
    Ls = msa["seq_aln"].str.len().unique()
    assert len(Ls) == 1, f"MSA nicht aligned: {Ls}"
    msa["seq_ungapped"] = msa["seq_aln"].str.replace("-","",regex=False).str.replace(".","",regex=False)
    log.info(f"MSA: {len(msa)} Sequenzen, aln_len={int(Ls[0])}")

    wt = wt_seq.replace("-","").replace(".","").strip()
    hits = msa.index[msa["seq_ungapped"] == wt].tolist()
    if hits:
        ref_idx = hits[0]
        log.info(f"Exakter WT Match: idx={ref_idx}")
    else:
        scores = [sum(a==b for a,b in zip(s, wt))/max(len(s),len(wt)) for s in msa["seq_ungapped"]]
        ref_idx = int(np.argmax(scores))
        log.info(f"Kein exakter Match. Bester: idx={ref_idx}, identity={scores[ref_idx]:.4f}")

    log.info(f"Ref: {msa.loc[ref_idx,'header'][:100]}")
    return msa, ref_idx


# ============================================================
# 3. Alignment kodieren + Positionen mappen
# ============================================================
GAP = ord("-")
DOT = ord(".")

def encode_aln(seqs):
    L = len(seqs[0])
    arr = np.empty((len(seqs), L), dtype=np.uint8)
    for i, s in enumerate(seqs):
        b = s.encode("ascii")
        assert len(b) == L
        arr[i] = np.frombuffer(b, dtype=np.uint8)
    return arr

def build_pos2aln(ref_aln_seq):
    m, pos = {}, 0
    for j, ch in enumerate(ref_aln_seq):
        if ch not in ["-", "."]:
            pos += 1
            m[pos] = j
    return m

def build_segment_positions(pg, segment_level, mutpos_col, pos2aln, ref_len):
    seg_pos_map = {}
    for seg_id, g in pg.groupby("_segment_id"):
        pos = set()
        for lst in g[mutpos_col]: pos.update(lst)
        pos = sorted([p for p in pos if 1 <= p <= ref_len and p in pos2aln])
        if pos:
            seg_pos_map[str(seg_id)] = pos
    return seg_pos_map


# ============================================================
# 4. Distanz-Hilfsfunktionen
# ============================================================
def distmat_on_alncols(arr, cols):
    sub = arr[:, cols]
    N = sub.shape[0]
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        si = sub[i]
        valid = (sub != GAP) & (si[None,:] != GAP) & (sub != DOT) & (si[None,:] != DOT)
        denom = valid.sum(axis=1).astype(np.float32)
        denom = np.where(denom == 0, 1.0, denom)
        mism  = ((sub != si[None,:]) & valid).sum(axis=1).astype(np.float32)
        D[i,:] = mism / denom
    np.fill_diagonal(D, 0.0)
    return D

def dist_to_ref_on_cols(arr, ref_vec, cols):
    sub     = arr[:, cols]
    ref     = ref_vec[cols]
    valid_r = (ref != GAP) & (ref != DOT)
    out = np.zeros(sub.shape[0], dtype=np.float32)
    for i in range(sub.shape[0]):
        s     = sub[i]
        valid = valid_r & (s != GAP) & (s != DOT)
        denom = int(valid.sum())
        out[i] = 1.0 if denom == 0 else float(((s != ref) & valid).sum()) / denom
    return out

def dist_to_ref_abs_on_cols(arr, ref_vec, cols):
    """Absolute Anzahl von Mismatches zur Referenz (ungekürzt)."""
    sub     = arr[:, cols]
    ref     = ref_vec[cols]
    valid_r = (ref != GAP) & (ref != DOT)
    out = np.zeros(sub.shape[0], dtype=np.int32)
    for i in range(sub.shape[0]):
        s     = sub[i]
        valid = valid_r & (s != GAP) & (s != DOT)
        out[i] = int(((s != ref) & valid).sum())
    return out

def encode_ungapped(seq):
    return np.frombuffer(seq.replace("-","").replace(".","").encode("ascii"), dtype=np.uint8)

def variant_segment_array(variant_seqs, seg_pos_ungapped):
    idx0 = np.array([p-1 for p in seg_pos_ungapped], dtype=np.int32)
    out  = np.empty((len(variant_seqs), len(idx0)), dtype=np.uint8)
    for i, s in enumerate(variant_seqs):
        out[i] = encode_ungapped(str(s))[idx0]
    return out

def nearest_orthologue_indices(variant_seqs, seg_pos_ungapped, aln_arr, pos2aln, chunk_size):
    aln_idx   = np.array([pos2aln[p] for p in seg_pos_ungapped], dtype=np.int32)
    ortho_seg = aln_arr[:, aln_idx]
    ortho_valid = (ortho_seg != GAP) & (ortho_seg != DOT)
    denom = ortho_valid.sum(axis=1).astype(np.float32)
    denom = np.where(denom == 0, 1.0, denom)
    out = np.empty(len(variant_seqs), dtype=np.int32)
    for start in range(0, len(variant_seqs), chunk_size):
        end   = min(start + chunk_size, len(variant_seqs))
        vseg  = variant_segment_array(variant_seqs[start:end], seg_pos_ungapped)
        mism  = (vseg[:,None,:] != ortho_seg[None,:,:]) & ortho_valid[None,:,:]
        dist  = mism.sum(axis=2).astype(np.float32) / denom[None,:]
        out[start:end] = dist.argmin(axis=1).astype(np.int32)
    return out


# ============================================================
# 5. Split-Logik
# ============================================================
def choose_clusters_by_distance(df, cluster_col, dist_col, target_fraction,
                                prefer_farthest=True, random_seed=67):
    g = (df.assign(_one=1)
           .groupby(cluster_col)
           .agg(n=("_one","sum"), dist_med=(dist_col,"median"))
           .reset_index())
    if len(g) == 0:
        return []
    g = g.sort_values("dist_med", ascending=not prefer_farthest) if prefer_farthest \
        else g.sample(frac=1.0, random_state=random_seed)
    total = len(df)
    need  = max(1, int(np.ceil(total * target_fraction)))
    picked, acc = [], 0
    for _, r in g.iterrows():
        picked.append(int(r[cluster_col]))
        acc += int(r["n"])
        if acc >= need:
            break
    return picked

def make_cluster_splits(df, cluster_col, dist_col,
                        test_fraction, val_fraction_within_train, random_seed=67):
    df = df.copy()
    test_clusters = choose_clusters_by_distance(
        df, cluster_col, dist_col, test_fraction, prefer_farthest=True,
        random_seed=random_seed)
    is_test    = df[cluster_col].isin(test_clusters)
    train_pool = df.loc[~is_test].copy()
    if len(train_pool) == 0:
        return pd.Series(["test"] * len(df), index=df.index)
    val_clusters = choose_clusters_by_distance(
        train_pool, cluster_col, dist_col, val_fraction_within_train,
        prefer_farthest=False, random_seed=random_seed)
    is_val = (~is_test) & df[cluster_col].isin(val_clusters)
    return pd.Series(
        np.where(is_test, "test", np.where(is_val, "val", "train")),
        index=df.index
    )


# ============================================================
# 6. Threshold-Sweep
# ============================================================
def run_sweep(pg, segment_positions, aln_arr, pos2aln, msa, ref_idx,
              seq_col, score_col, test_fraction, val_fraction_within_train,
              chunk_size, func_threshold, random_seed):
    msa_oid    = msa["id"].astype(str).to_numpy()
    msa_header = msa["header"].astype(str).to_numpy()
    ref_vec    = aln_arr[ref_idx]

    sweep_rows = []

    for seg_id, seg_pos in sorted(segment_positions.items()):
        seg_df    = pg.loc[pg["_segment_id"] == seg_id].copy()
        n_variants = len(seg_df)
        n_pos     = len(seg_pos)
        if n_variants < 200 or n_pos == 0:
            log.info(f"  {seg_id}: SKIP (n_variants={n_variants}, n_pos={n_pos})")
            continue

        cols = np.array([pos2aln[p] for p in seg_pos], dtype=np.int32)
        Dseg = distmat_on_alncols(aln_arr, cols)
        Zseg = linkage(squareform(Dseg, checks=False), method="average")
        distref_seg = dist_to_ref_on_cols(aln_arr, ref_vec, cols)

        nearest_idx = nearest_orthologue_indices(
            seg_df[seq_col].tolist(), seg_pos, aln_arr, pos2aln, chunk_size)

        seg_df["nearest_orthologue_id"]     = msa_oid[nearest_idx]
        seg_df["nearest_orthologue_header"] = msa_header[nearest_idx]
        seg_df["nearest_dist_to_ref"]       = distref_seg[nearest_idx]

        log.info(f"  {seg_id}: {n_variants} Varianten, {n_pos} Positionen → Sweep ...")

        for thr in THRESHOLDS:
            species_cluster = fcluster(Zseg, t=thr, criterion="distance")
            seg_df["nearest_species_cluster"] = species_cluster[nearest_idx]

            seg_df["split"] = make_cluster_splits(
                seg_df, "nearest_species_cluster", "nearest_dist_to_ref",
                test_fraction, val_fraction_within_train, random_seed)

            vc      = seg_df["split"].value_counts().to_dict()
            n_train = int(vc.get("train", 0))
            n_val   = int(vc.get("val",   0))
            n_test  = int(vc.get("test",  0))

            tr_cl = set(seg_df.loc[seg_df["split"]=="train", "nearest_species_cluster"])
            va_cl = set(seg_df.loc[seg_df["split"]=="val",   "nearest_species_cluster"])
            te_cl = set(seg_df.loc[seg_df["split"]=="test",  "nearest_species_cluster"])
            disjoint_ok = (len(tr_cl & va_cl) == 0) and (len(tr_cl & te_cl) == 0) and (len(va_cl & te_cl) == 0)

            med         = seg_df.groupby("split")["nearest_dist_to_ref"].median().to_dict()
            med_train   = float(med.get("train", np.nan))
            med_test    = float(med.get("test",  np.nan))
            ood_gap     = med_test - med_train if (np.isfinite(med_test) and np.isfinite(med_train)) else np.nan

            s_train = seg_df.loc[seg_df["split"]=="train", score_col].dropna()
            s_test  = seg_df.loc[seg_df["split"]=="test",  score_col].dropna()
            s_val   = seg_df.loc[seg_df["split"]=="val",   score_col].dropna()

            frac_func_train = float((s_train > func_threshold).mean()) if len(s_train) > 0 else np.nan
            frac_func_test  = float((s_test  > func_threshold).mean()) if len(s_test)  > 0 else np.nan
            frac_func_val   = float((s_val   > func_threshold).mean()) if len(s_val)   > 0 else np.nan
            mean_dms_train  = float(s_train.mean()) if len(s_train) > 0 else np.nan
            mean_dms_test   = float(s_test.mean())  if len(s_test)  > 0 else np.nan
            emd_train_test  = float(wasserstein_distance(s_train.values, s_test.values)) \
                              if (len(s_train) > 0 and len(s_test) > 0) else np.nan

            sweep_rows.append({
                "segment_id":        seg_id,
                "threshold":         thr,
                "n_variants":        n_variants,
                "n_positions":       n_pos,
                "n_clusters_used":   seg_df["nearest_species_cluster"].nunique(),
                "n_train":           n_train,
                "n_val":             n_val,
                "n_test":            n_test,
                "median_dist_train": med_train,
                "median_dist_test":  med_test,
                "ood_gap":           ood_gap,
                "disjoint_ok":       disjoint_ok,
                "frac_func_train":   frac_func_train,
                "frac_func_test":    frac_func_test,
                "frac_func_val":     frac_func_val,
                "mean_dms_train":    mean_dms_train,
                "mean_dms_test":     mean_dms_test,
                "emd_train_test":    emd_train_test,
            })

    return pd.DataFrame(sweep_rows)


# ============================================================
# 7. Beste Kandidaten auswählen
# ============================================================
def _score(df):
    """Composite score: hoher OOD-Gap + niedrige EMD + funktionale Test-Varianten + Trainmenge."""
    return (
        df["ood_gap"].rank(pct=True)                    * 0.30 +
        (1 - df["emd_train_test"].rank(pct=True))       * 0.35 +
        df["frac_func_test"].rank(pct=True)             * 0.25 +
        df["n_train"].rank(pct=True)                    * 0.10
    )

def _pick_best(df):
    return (
        df.sort_values("score", ascending=False)
          .groupby("segment_id")
          .head(1)
          .sort_values("score", ascending=False)
          .reset_index(drop=True)
    )

def select_best_per_segment(sweep, min_train, min_test, max_test_fraction,
                            max_val_fraction, min_ood_gap, relaxed_fallback=True):
    s = sweep.copy()
    s["test_frac"] = s["n_test"] / s["n_variants"]
    s["val_frac"]  = s["n_val"]  / s["n_variants"]

    # --- Strenger Filter ---
    ok = s[
        s["disjoint_ok"] &
        (s["n_train"]   >= min_train) &
        (s["n_test"]    >= min_test) &
        (s["test_frac"] <= max_test_fraction) &
        (s["val_frac"]  <= max_val_fraction) &
        (s["ood_gap"]   >  min_ood_gap)
    ].copy()

    log.info(f"Sweep-Kandidaten nach Grundfilter: {len(ok)}")

    if len(ok) > 0:
        ok["score"]    = _score(ok)
        ok["fallback"] = False
    else:
        ok = pd.DataFrame()

    best_per_seg = _pick_best(ok) if len(ok) > 0 else pd.DataFrame()

    # --- Relaxed Fallback für Segmente ohne strikten Split ---
    if relaxed_fallback:
        covered = set(best_per_seg["segment_id"]) if len(best_per_seg) > 0 else set()
        remaining = s[~s["segment_id"].isin(covered)].copy()

        if len(remaining) > 0:
            # Gelockerte Bedingungen:
            # - kein max_test_fraction / max_val_fraction
            # - min_train und min_test deutlich kleiner
            # - ood_gap > 0 (irgendein OOD-Signal)
            # - Unter mehreren Thresholds: wähle den mit test_frac am nächsten an Ziel
            relaxed_min_train = max(100, min_train // 10)
            relaxed_min_test  = max(50,  min_test  // 10)

            ok2 = remaining[
                remaining["disjoint_ok"] &
                (remaining["n_train"] >= relaxed_min_train) &
                (remaining["n_test"]  >= relaxed_min_test) &
                (remaining["ood_gap"] >= 0)
            ].copy()

            if len(ok2) > 0:
                # Score: bevorzuge test_frac nahe am Ziel + OOD-Gap
                ok2["frac_dev"]  = (ok2["test_frac"] - max_test_fraction * 0.4).abs()
                ok2["score"]     = (
                    ok2["ood_gap"].rank(pct=True)           * 0.50 +
                    (1 - ok2["frac_dev"].rank(pct=True))    * 0.30 +
                    ok2["n_train"].rank(pct=True)           * 0.20
                )
                ok2["fallback"] = True

                fallback_best = _pick_best(ok2)
                log.info(f"Relaxed Fallback: {len(fallback_best)} weitere Segmente "
                         f"({', '.join(fallback_best['segment_id'].tolist())})")
                for _, r in fallback_best.iterrows():
                    log.info(f"  {r['segment_id']:6s} thr={r['threshold']:.2f} | "
                             f"train={r['n_train']:5d}  val={r['n_val']:4d}  "
                             f"test={r['n_test']:5d}  test_frac={r['test_frac']:.2f}  "
                             f"ood_gap={r['ood_gap']:.3f}  [FALLBACK]")

                best_per_seg = pd.concat([best_per_seg, fallback_best], ignore_index=True)
            else:
                not_covered = set(remaining["segment_id"].unique()) - set()
                log.info(f"Kein Fallback-Kandidat für: {sorted(not_covered)}")

    if len(best_per_seg) == 0:
        log.warning("Keine Kandidaten gefunden (weder streng noch relaxed).")

    return best_per_seg


# ============================================================
# 8. Splits speichern
# ============================================================
def save_segment_splits(pg, best_per_seg, segment_positions, aln_arr, pos2aln,
                        msa, ref_idx, seq_col, score_col,
                        test_fraction, val_fraction_within_train, chunk_size,
                        random_seed, out_dir, segment_level):
    msa_oid    = msa["id"].astype(str).to_numpy()
    msa_header = msa["header"].astype(str).to_numpy()
    ref_vec    = aln_arr[ref_idx]

    all_dfs        = []
    saved_segments = []

    for _, row in best_per_seg.iterrows():
        seg_id  = str(row["segment_id"])
        thr     = float(row["threshold"])
        seg_pos = segment_positions[seg_id]
        seg_df  = pg.loc[pg["_segment_id"] == seg_id].copy()
        cols    = np.array([pos2aln[p] for p in seg_pos], dtype=np.int32)

        Dseg = distmat_on_alncols(aln_arr, cols)
        Zseg = linkage(squareform(Dseg, checks=False), method="average")
        species_cluster = fcluster(Zseg, t=thr, criterion="distance")
        distref_seg     = dist_to_ref_on_cols(aln_arr, ref_vec, cols)

        nearest_idx = nearest_orthologue_indices(
            seg_df[seq_col].tolist(), seg_pos, aln_arr, pos2aln, chunk_size)

        seg_df["nearest_orthologue_id"]     = msa_oid[nearest_idx]
        seg_df["nearest_orthologue_header"] = msa_header[nearest_idx]
        seg_df["nearest_species_cluster"]   = species_cluster[nearest_idx]
        seg_df["nearest_dist_to_ref"]       = distref_seg[nearest_idx]

        seg_df["split"] = make_cluster_splits(
            seg_df, "nearest_species_cluster", "nearest_dist_to_ref",
            test_fraction, val_fraction_within_train, random_seed)

        # Tausche train↔val wenn val größer als train ist
        vc = seg_df["split"].value_counts()
        if vc.get("val", 0) > vc.get("train", 0):
            seg_df["split"] = seg_df["split"].replace({"train": "val", "val": "train"})
            log.info(f"  {seg_id:6s}: train↔val getauscht (val war größer als train)")
            vc = seg_df["split"].value_counts()

        is_fb    = bool(row.get("fallback", False))
        fb_label = "  [FALLBACK]" if is_fb else ""
        log.info(f"  {seg_id:6s} thr={thr:.2f} | "
                 f"train={vc.get('train',0):5d}  val={vc.get('val',0):4d}  "
                 f"test={vc.get('test',0):5d}{fb_label}")

        # Cluster-Diagnose
        msa_cluster_sizes = pd.Series(species_cluster).value_counts().sort_index()
        n_clusters = len(msa_cluster_sizes)
        n_singleton = int((msa_cluster_sizes == 1).sum())
        log.info(f"  {seg_id:6s} Cluster: {n_clusters} total "
                 f"(davon {n_singleton} Singletons = 1 Spezies) | "
                 f"MSA-Sequenzen pro Cluster (min/median/max): "
                 f"{msa_cluster_sizes.min()} / {msa_cluster_sizes.median():.0f} / {msa_cluster_sizes.max()}")
        cluster_split = (seg_df.groupby("nearest_species_cluster")["split"]
                         .value_counts().unstack(fill_value=0))
        for col in ["train", "val", "test"]:
            if col not in cluster_split.columns:
                cluster_split[col] = 0
        cluster_split = cluster_split[["train", "val", "test"]]
        cluster_split["total"] = cluster_split.sum(axis=1)
        n_cl_train = int((cluster_split["train"] > 0).sum())
        n_cl_val   = int((cluster_split["val"]   > 0).sum())
        n_cl_test  = int((cluster_split["test"]  > 0).sum())
        log.info(f"  {seg_id:6s} Cluster pro Split: "
                 f"train={n_cl_train}  val={n_cl_val}  test={n_cl_test}  "
                 f"(von {n_clusters} gesamt)")
        log.info(f"  {seg_id:6s} Cluster→Split-Verteilung (Varianten):\n"
                 + cluster_split.to_string())

        base = os.path.join(out_dir, f"{segment_level}_{seg_id}_thr{thr}")
        seg_df.to_csv(base + "_all.csv",   index=False)
        seg_df.loc[seg_df["split"]=="train"].to_csv(base + "_train.csv", index=False)
        seg_df.loc[seg_df["split"]=="val"  ].to_csv(base + "_val.csv",   index=False)
        seg_df.loc[seg_df["split"]=="test" ].to_csv(base + "_test.csv",  index=False)

        saved_segments.append(seg_id)
        all_dfs.append(seg_df)

    # Debug: fehlende Segmente
    all_seg_ids     = set(pg["_segment_id"].dropna().unique())
    missing_seg_ids = all_seg_ids - set(saved_segments)
    n_missing       = int(pg["_segment_id"].isin(missing_seg_ids).sum())

    log.info(f"\n=== {len(missing_seg_ids)} Segmente OHNE Split "
             f"(zu wenig Daten oder schlechte OOD-Struktur) ===")
    for s in sorted(missing_seg_ids):
        n = len(pg[pg["_segment_id"] == s])
        log.info(f"  {s}: {n} Varianten")
    log.info(f"Varianten ohne Split: {n_missing}/{len(pg)} ({n_missing/len(pg)*100:.1f}%)")

    return all_dfs, saved_segments


# ============================================================
# 9. Kombinierter Split
# ============================================================
def save_combined_split(all_dfs, out_dir, segment_level):
    combined = pd.concat(all_dfs, ignore_index=True)
    vc       = combined["split"].value_counts()

    log.info("\n=== Kombinierter Split (alle Segmente) ===")
    log.info(f"  {vc.to_dict()}")
    log.info(f"  Segmente: {combined['_segment_id'].nunique()}, Gesamt: {len(combined):,}")

    base = os.path.join(out_dir, f"{segment_level}_ALL_combined")
    combined.to_csv(base + "_all.csv",   index=False)
    combined.loc[combined["split"]=="train"].to_csv(base + "_train.csv", index=False)
    combined.loc[combined["split"]=="val"  ].to_csv(base + "_val.csv",   index=False)
    combined.loc[combined["split"]=="test" ].to_csv(base + "_test.csv",  index=False)
    log.info(f"  Gespeichert: {base}_{{all,train,val,test}}.csv")

    # Tabelle pro Segment
    pivot = combined.groupby(["_segment_id", "split"]).size().unstack(fill_value=0)
    for col in ["train", "val", "test"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["train", "val", "test"]]
    pivot["total"]     = pivot.sum(axis=1)
    pivot["test_frac"] = (pivot["test"] / pivot["total"]).round(3)
    log.info("\n" + pivot.to_string())

    return combined


# ============================================================
# 10. Optional: Diagnose-Plots
# ============================================================
def make_plots(best_per_seg, pg, segment_positions, aln_arr, pos2aln,
               msa, ref_idx, seq_col, score_col,
               test_fraction, val_fraction_within_train, chunk_size,
               random_seed, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_vec    = aln_arr[ref_idx]
    n_seg      = len(best_per_seg)
    if n_seg == 0:
        return

    fig, axes = plt.subplots(n_seg, 3, figsize=(21, 4 * n_seg))
    if n_seg == 1:
        axes = axes.reshape(1, -1)
    colors = {"train": "#2196F3", "val": "#FF9800", "test": "#F44336"}

    for i, (_, row) in enumerate(best_per_seg.iterrows()):
        seg_id  = str(row["segment_id"])
        thr     = float(row["threshold"])
        seg_pos = segment_positions[seg_id]
        seg_df  = pg.loc[pg["_segment_id"] == seg_id].copy()
        cols    = np.array([pos2aln[p] for p in seg_pos], dtype=np.int32)

        Dseg            = distmat_on_alncols(aln_arr, cols)
        Zseg            = linkage(squareform(Dseg, checks=False), method="average")
        species_cluster = fcluster(Zseg, t=thr, criterion="distance")
        distref_seg     = dist_to_ref_on_cols(aln_arr, ref_vec, cols)
        distref_abs_seg = dist_to_ref_abs_on_cols(aln_arr, ref_vec, cols)
        nearest_idx     = nearest_orthologue_indices(
            seg_df[seq_col].tolist(), seg_pos, aln_arr, pos2aln, chunk_size)

        seg_df["nearest_species_cluster"]   = species_cluster[nearest_idx]
        seg_df["nearest_dist_to_ref"]       = distref_seg[nearest_idx]
        seg_df["nearest_dist_to_ref_abs"]   = distref_abs_seg[nearest_idx]
        seg_df["split"] = make_cluster_splits(
            seg_df, "nearest_species_cluster", "nearest_dist_to_ref",
            test_fraction, val_fraction_within_train, random_seed)

        rng_norm = (seg_df["nearest_dist_to_ref"].min(),     seg_df["nearest_dist_to_ref"].max())
        rng_abs  = (seg_df["nearest_dist_to_ref_abs"].min(), seg_df["nearest_dist_to_ref_abs"].max())
        rng_score= (seg_df[score_col].min(),                 seg_df[score_col].max())

        # Sort splits so the smallest bar is drawn last (= rendered in front)
        splits_by_size = sorted(
            ["train", "val", "test"],
            key=lambda sp: len(seg_df.loc[seg_df["split"] == sp]),
            reverse=True,
        )

        ax = axes[i, 0]
        for sp in splits_by_size:
            vals = seg_df.loc[seg_df["split"]==sp, "nearest_dist_to_ref"]
            if len(vals) > 0:
                ax.hist(vals, bins=30, range=rng_norm, alpha=0.6, label=f"{sp} (n={len(vals)})", color=colors[sp])
        ax.set_title(f"{seg_id} thr={thr}: Distance to Reference (normalised)")
        ax.set_xlabel("Hamming Distance (fraction)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

        ax = axes[i, 1]
        for sp in splits_by_size:
            vals = seg_df.loc[seg_df["split"]==sp, "nearest_dist_to_ref_abs"]
            if len(vals) > 0:
                ax.hist(vals, bins=30, range=rng_abs, alpha=0.6, label=f"{sp} (n={len(vals)})", color=colors[sp])
        ax.set_title(f"{seg_id} thr={thr}: Distance to Reference (absolute, {len(cols)} pos.)")
        ax.set_xlabel("Number of mismatches")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

        ax = axes[i, 2]
        for sp in splits_by_size:
            vals = seg_df.loc[seg_df["split"]==sp, score_col]
            if len(vals) > 0:
                ax.hist(vals, bins=50, range=rng_score, alpha=0.6, label=f"{sp} (n={len(vals)})", color=colors[sp])
        ax.set_title(f"{seg_id} thr={thr}: DMS Score")
        ax.set_xlabel("DMS Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(out_dir, "segment_splits_overview.png")
    plt.savefig(out, dpi=150)
    log.info(f"Plot saved: {out}")


# ============================================================
# main
# ============================================================
def main():
    args = parse_args()
    rng  = np.random.default_rng(args.random_seed)   # noqa: F841 (for reproducibility)
    os.makedirs(args.out_dir, exist_ok=True)

    log.info("=== HIS3 Phylogenetic Species-Cluster Split ===")

    # 1. Daten
    pg = load_pg(
        args.csv_path, args.segment_level,
        seq_col=DEFAULTS["seq_col"],
        mutpos_col=DEFAULTS["mutpos_col"],
        segment_type_col=DEFAULTS["segment_type_col"],
        allowed_types=DEFAULTS["allowed_types"],
    )

    # 2. MSA
    msa, ref_idx = load_msa(args.fasta_path, args.wt_seq)

    # 3. Alignment kodieren
    aln_arr = encode_aln(msa["seq_aln"].astype(str).tolist())
    ref_aln = msa.loc[ref_idx, "seq_aln"]
    pos2aln = build_pos2aln(ref_aln)
    ref_len = max(pos2aln.keys())
    log.info(f"Ref ungapped len: {ref_len}, aln len: {aln_arr.shape[1]}")

    segment_positions = build_segment_positions(
        pg, args.segment_level, DEFAULTS["mutpos_col"], pos2aln, ref_len)
    log.info(f"{len(segment_positions)} Segmente mit validen Positionen:")
    for s in sorted(segment_positions):
        pos = segment_positions[s]
        n   = len(pg[pg["_segment_id"] == s])
        log.info(f"  {s}: {n} Varianten, {len(pos)} Pos ({min(pos)}-{max(pos)})")

    # 4. Sweep
    log.info("\n=== Threshold-Sweep ===")
    sweep = run_sweep(
        pg, segment_positions, aln_arr, pos2aln, msa, ref_idx,
        seq_col=DEFAULTS["seq_col"], score_col=DEFAULTS["score_col"],
        test_fraction=args.test_fraction,
        val_fraction_within_train=args.val_fraction_within_train,
        chunk_size=args.chunk_size,
        func_threshold=args.func_threshold,
        random_seed=args.random_seed,
    )
    sweep.to_csv(os.path.join(args.out_dir, "sweep_results.csv"), index=False)
    log.info(f"Sweep fertig: {len(sweep)} Zeilen → sweep_results.csv")

    # 5. Beste Kandidaten
    best_per_seg = select_best_per_segment(
        sweep, args.min_train, args.min_test,
        args.max_test_fraction, args.max_val_fraction, args.min_ood_gap,
        relaxed_fallback=not args.no_relaxed_fallback)

    if len(best_per_seg) == 0:
        log.error("Keine geeigneten Kandidaten gefunden. Fertig.")
        return

    log.info("\n=== Bester Threshold pro Segment ===")
    display_cols = ["segment_id", "threshold", "n_train", "n_val", "n_test",
                    "ood_gap", "frac_func_train", "frac_func_test",
                    "mean_dms_train", "mean_dms_test", "emd_train_test", "score"]
    log.info("\n" + best_per_seg[display_cols].to_string(index=False))
    best_per_seg.to_csv(os.path.join(args.out_dir, "best_per_segment.csv"), index=False)

    # 6. Per-Segment Splits speichern
    log.info("\n=== Speichere per-Segment Splits ===")
    all_dfs, saved_segments = save_segment_splits(
        pg, best_per_seg, segment_positions, aln_arr, pos2aln, msa, ref_idx,
        seq_col=DEFAULTS["seq_col"], score_col=DEFAULTS["score_col"],
        test_fraction=args.test_fraction,
        val_fraction_within_train=args.val_fraction_within_train,
        chunk_size=args.chunk_size,
        random_seed=args.random_seed,
        out_dir=args.out_dir,
        segment_level=args.segment_level,
    )

    # 7. Kombinierter Split
    log.info("\n=== Speichere kombinierten Split ===")
    save_combined_split(all_dfs, args.out_dir, args.segment_level)

    # 8. Plots
    if args.plot:
        log.info("\n=== Erzeuge Diagnose-Plots ===")
        make_plots(
            best_per_seg, pg, segment_positions, aln_arr, pos2aln, msa, ref_idx,
            seq_col=DEFAULTS["seq_col"], score_col=DEFAULTS["score_col"],
            test_fraction=args.test_fraction,
            val_fraction_within_train=args.val_fraction_within_train,
            chunk_size=args.chunk_size,
            random_seed=args.random_seed,
            out_dir=args.out_dir,
        )

    log.info("\n=== Fertig! ===")
    log.info(f"Output in: {args.out_dir}")
    for f in sorted(os.listdir(args.out_dir)):
        if f.endswith(".csv"):
            n = sum(1 for _ in open(os.path.join(args.out_dir, f))) - 1
            log.info(f"  {f}  ({n:,} rows)")


if __name__ == "__main__":
    main()
