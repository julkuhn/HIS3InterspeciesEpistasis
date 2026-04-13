#!/usr/bin/env python3
"""
Extended correlation analysis:
  Fig A — Segment-level: bimodality, score skewness, ICC, cluster gap
  Fig B — Per-cluster: ICC, within-cluster score range, nearest-train-distance
  Fig C — Mutation position heatmap (where in protein are test errors highest?)
  Fig D — Correlation matrix heatmap across all segment-level features
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

BASE = Path("/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra")
OUT  = BASE / "metl_rosetta"

SEG_THR  = {"S02":"thr0.08","S03":"thr0.1","S04":"thr0.08","S05":"thr0.25",
            "S06":"thr0.18","S07":"thr0.05","S08":"thr0.18","S12":"thr0.1"}
METL_V1  = {"S02":"XGWfEkoP","S03":"bat4JTEc","S04":"e74qnHH5","S05":"f6ct8j7L",
             "S06":"F7uG4XHT","S07":"WkqJQ2ni","S08":"RCd8d3kN","S12":"N5SHB9oU"}
COLORS   = {"ESM":"#2196F3","METL-v1":"#FF9800","METL-v2":"#E91E63"}
SEQ_LEN  = 220

def load_esm(seg):
    d = BASE/f"esm_results/{seg}"
    with open(d/"mlp_model.pkl","rb") as f: m = pickle.load(f)
    with open(d/"scaler.pkl","rb")    as f: s = pickle.load(f)
    return m.predict(s.transform(np.load(d/"X_test.npy")))

def load_mv1(seg):
    return np.load(BASE/f"metl_rosetta/target_model_45k/{METL_V1[seg]}/predictions/test_predictions.npy").squeeze()

def find_mv2(seg):
    for d in (BASE/"metl_rosetta/target_model_45k_v2").iterdir():
        hf = d/"hparams.yaml"; pf = d/"predictions/test_predictions.npy"
        if hf.exists() and f"his3_{seg}_local_45k_v2" in open(hf).read() and pf.exists():
            return np.load(pf).squeeze()
    return None

def get_test(seg, thr):
    df = pd.read_csv(BASE/f"splits_segmentwise_species/super_segments_{seg}_{thr}_all.csv")
    idx = np.loadtxt(BASE/f"metl_rosetta/metl_splits/{seg}_{thr}/test.txt", dtype=int)
    return df, df.iloc[idx].reset_index(drop=True)

def seg_features(seg, thr):
    df_all, te = get_test(seg, thr)
    tr = df_all[df_all.split=='train']
    y  = te['DMS_score'].values
    n  = len(te)
    skew = float(te['DMS_score'].skew())
    kurt = float(te['DMS_score'].kurt())
    bc   = (skew**2+1)/(kurt+3*(n-1)**2/((n-2)*(n-3)))
    # min dist between test and train clusters
    tc = te.groupby('nearest_species_cluster')['nearest_dist_to_ref'].mean()
    tr_c = tr.groupby('nearest_species_cluster')['nearest_dist_to_ref'].mean()
    min_gap = float(abs(tc.values[:,None]-tr_c.values[None,:]).min())
    # ICC
    bv = float(te.groupby('nearest_species_cluster')['DMS_score'].mean().var())
    wv = float(te.groupby('nearest_species_cluster')['DMS_score'].var().mean())
    icc = bv/(bv+wv) if (bv+wv)>0 else 0
    return dict(
        n_train    = int((df_all.split=='train').sum()),
        frac_func  = float(te['DMS_score_bin'].mean()),
        score_std  = float(y.std()),
        score_range= float(y.max()-y.min()),
        n_mut_mean = float(te['n_mut'].mean()),
        dist_delta = float(te['nearest_dist_to_ref'].mean() - tr['nearest_dist_to_ref'].mean()),
        bimodality = float(bc),
        skewness   = float(abs(skew)),   # abs so high=more extreme
        icc        = float(icc),
        min_clust_gap = float(min_gap),
        n_test_clusters = int(te['nearest_species_cluster'].nunique()),
    )

# ── build master table ───────────────────────────────────────────────────────
rows = []
for seg, thr in SEG_THR.items():
    feats = seg_features(seg, thr)
    _, te  = get_test(seg, thr)
    y      = te['DMS_score'].values
    p_esm  = load_esm(seg)
    p_mv1  = load_mv1(seg)
    p_mv2  = find_mv2(seg)
    rows.append({**feats,
        'seg':    seg,
        'sp_esm': stats.spearmanr(p_esm, y).statistic,
        'sp_mv1': stats.spearmanr(p_mv1, y).statistic,
        'sp_mv2': stats.spearmanr(p_mv2, y).statistic if p_mv2 is not None else np.nan,
        'p_esm': p_esm, 'p_mv1': p_mv1, 'p_mv2': p_mv2, 'y': y, 'te': te,
    })

sdf = pd.DataFrame([{k:v for k,v in r.items()
                      if k not in ('p_esm','p_mv1','p_mv2','y','te')} for r in rows])

# ── Figure A: new factors (bimodality, skewness, ICC, cluster gap) ───────────
new_factors = [
    ("bimodality",     "Score bimodality\n(>0.55 = bimodal)"),
    ("skewness",       "|Skewness| of score\ndistribution"),
    ("icc",            "ICC: between-cluster\nscore variance fraction"),
    ("min_clust_gap",  "Min dist: test cluster\nto nearest train cluster"),
    ("n_test_clusters","Number of test\nspecies clusters"),
]

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for ax, (fcol, flabel) in zip(axes, new_factors):
    x = sdf[fcol].values
    for model, col, c, mk in [("ESM","sp_esm",COLORS["ESM"],"o"),
                                ("METL-v1","sp_mv1",COLORS["METL-v1"],"^"),
                                ("METL-v2","sp_mv2",COLORS["METL-v2"],"D")]:
        yv = sdf[col].values
        valid = ~np.isnan(yv)
        rho, p = stats.spearmanr(x[valid], yv[valid])
        ax.scatter(x[valid], yv[valid], color=c, marker=mk, s=90, zorder=5,
                   label=f"{model} ρ={rho:+.2f} p={p:.2f}")
        if valid.sum() > 2:
            m, b = np.polyfit(x[valid], yv[valid], 1)
            xf = np.linspace(x.min(), x.max(), 100)
            ax.plot(xf, m*xf+b, color=c, lw=1.5, alpha=0.6)
    for _, r in sdf.iterrows():
        ax.annotate(r.seg, (r[fcol], r.sp_esm), xytext=(3,3),
                    textcoords="offset points", fontsize=7.5, color=COLORS["ESM"])
    ax.set_xlabel(flabel, fontsize=10)
    ax.set_ylabel("Spearman ρ (OOD test)" if fcol==new_factors[0][0] else "", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.0)

fig.suptitle("Extended factor analysis: bimodality, ICC, cluster gap\n(n=8 segments)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT/"plot_extended_factors.png", dpi=150, bbox_inches="tight")
print("Saved plot_extended_factors.png")
plt.close()

# ── Figure B: Full correlation matrix ────────────────────────────────────────
feat_cols = ["n_train","frac_func","score_std","score_range","n_mut_mean",
             "dist_delta","bimodality","skewness","icc","min_clust_gap","n_test_clusters",
             "sp_esm","sp_mv1","sp_mv2"]
feat_labels = ["n_train","frac_func","score_std","score_range","n_mut_mean",
               "Δdist","bimodality","|skewness|","ICC","min_clust_gap","n_clusters",
               "ESM ρ","METL-v1 ρ","METL-v2 ρ"]

# Spearman correlation matrix
corr_mat = np.zeros((len(feat_cols), len(feat_cols)))
pval_mat = np.zeros((len(feat_cols), len(feat_cols)))
for i, c1 in enumerate(feat_cols):
    for j, c2 in enumerate(feat_cols):
        x1 = sdf[c1].values; x2 = sdf[c2].values
        valid = ~(np.isnan(x1)|np.isnan(x2))
        if valid.sum() > 2:
            r, p = stats.spearmanr(x1[valid], x2[valid])
            corr_mat[i,j] = r; pval_mat[i,j] = p
        else:
            corr_mat[i,j] = np.nan

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
ax.set_xticks(range(len(feat_labels))); ax.set_xticklabels(feat_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(feat_labels))); ax.set_yticklabels(feat_labels, fontsize=9)
# Annotate with values; mark significant (p<0.2 given n=8) with *
for i in range(len(feat_cols)):
    for j in range(len(feat_cols)):
        v = corr_mat[i,j]
        if np.isnan(v): continue
        sig = "*" if pval_mat[i,j] < 0.2 else ""
        ax.text(j, i, f"{v:.2f}{sig}", ha="center", va="center",
                fontsize=7, color="white" if abs(v)>0.6 else "black")
# Highlight performance rows
for idx in [-3,-2,-1]:
    ax.axhline(len(feat_cols)+idx+0.5, color="black", lw=1.5)
    ax.axvline(len(feat_cols)+idx+0.5, color="black", lw=1.5)

ax.set_title("Spearman correlation matrix — all segment-level features\n(* = p<0.2, n=8 segments)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT/"plot_correlation_matrix.png", dpi=150, bbox_inches="tight")
print("Saved plot_correlation_matrix.png")
plt.close()

# ── Figure C: Mutation position → prediction error heatmap ──────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 9))

for row_i, seg in enumerate(["S02","S08"]):
    r   = next(x for x in rows if x['seg']==seg)
    te  = r['te']; y = r['y']

    # Parse mutation positions
    import ast
    positions = te['mut_pos'].apply(lambda x: ast.literal_eval(x) if isinstance(x,str) else x)

    # Per-position error accumulator
    for col_i, (model, pred, c) in enumerate([
        ("ESM",     r['p_esm'], COLORS["ESM"]),
        ("METL-v1", r['p_mv1'], COLORS["METL-v1"]),
        ("METL-v2", r['p_mv2'], COLORS["METL-v2"]),
    ]):
        ax = axes[row_i][col_i]
        if pred is None:
            ax.text(0.5,0.5,"N/A",ha="center",va="center",transform=ax.transAxes)
            continue

        err = np.abs(pred - y)
        # Count occurrences and mean error per position
        pos_err   = np.zeros(SEQ_LEN)
        pos_count = np.zeros(SEQ_LEN)
        for i, (plist, e) in enumerate(zip(positions, err)):
            for p in plist:
                if 0 <= p < SEQ_LEN:
                    pos_err[p]   += e
                    pos_count[p] += 1

        # Running window mean error (smooth over 5 positions)
        with np.errstate(invalid="ignore"):
            mean_err = np.where(pos_count>0, pos_err/pos_count, np.nan)
        # smoothed
        from numpy.lib.stride_tricks import sliding_window_view
        w=7
        smoothed = np.full(SEQ_LEN, np.nan)
        for i in range(w//2, SEQ_LEN-w//2):
            window = mean_err[i-w//2:i+w//2+1]
            valid  = ~np.isnan(window)
            if valid.sum() >= 3:
                smoothed[i] = window[valid].mean()

        x_pos = np.arange(SEQ_LEN)
        ax.bar(x_pos, np.nan_to_num(mean_err), width=1,
               color=c, alpha=0.3, label="per-position mean |err|")
        ax.plot(x_pos, smoothed, color=c, lw=2, label="7-pos smoothed")

        # Overlay: position coverage (how often each pos appears)
        ax2 = ax.twinx()
        ax2.plot(x_pos, pos_count, color="gray", lw=0.8, alpha=0.5, ls="--")
        ax2.set_ylabel("Mutation count", color="gray", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)

        # Spearman: position_count vs mean_err
        valid_pos = (~np.isnan(mean_err)) & (pos_count > 5)
        if valid_pos.sum() > 5:
            rho, pv = stats.spearmanr(pos_count[valid_pos], mean_err[valid_pos])
        else:
            rho, pv = 0, 1

        ax.set_xlabel("Sequence position (0-indexed)", fontsize=9)
        ax.set_ylabel("|Prediction error|" if col_i==0 else "", fontsize=9)
        ax.set_title(f"{seg} — {model}\ncoverage vs err ρ={rho:+.2f}", fontsize=9.5)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(0, SEQ_LEN)
        ax.grid(True, alpha=0.2, axis="y")

fig.suptitle("Prediction error by sequence position\n"
             "(are certain protein regions harder to predict?)",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT/"plot_position_error.png", dpi=150, bbox_inches="tight")
print("Saved plot_position_error.png")
plt.close()

# ── Print summary of strongest correlations ──────────────────────────────────
print("\n=== All Spearman correlations with ESM performance (sorted by |rho|) ===")
pred_idx = feat_cols.index("sp_esm")
results = []
for i, (fc, fl) in enumerate(zip(feat_cols, feat_labels)):
    if fc in ("sp_esm","sp_mv1","sp_mv2"): continue
    rho = corr_mat[pred_idx, i]
    p   = pval_mat[pred_idx, i]
    results.append((abs(rho), rho, p, fl))
for _, rho, p, fl in sorted(results, reverse=True):
    sig = "**" if p<0.1 else ("*" if p<0.2 else "")
    print(f"  {fl:<20}: rho={rho:+.3f}  p={p:.3f}  {sig}")
print("\nDone.")
