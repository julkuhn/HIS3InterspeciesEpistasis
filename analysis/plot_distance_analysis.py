#!/usr/bin/env python3
"""
Two-panel analysis:
  1. Model performance vs. train/test phylogenetic distance (per segment)
  2. Per-cluster Spearman vs. distance for S02 and S08
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

# ── colour / style ───────────────────────────────────────────────────────────
MODEL_STYLE = {
    "ESM-2 650M":      dict(color="#2196F3", marker="o", ls="-",  lw=2),
    "MSA-T 100M":      dict(color="#4CAF50", marker="s", ls="--", lw=2),
    "METL-Local v1":   dict(color="#FF9800", marker="^", ls="-.", lw=2),
    "METL-Local v2":   dict(color="#E91E63", marker="D", ls=":",  lw=2),
}

SEGMENTS = ["S02","S03","S04","S05","S06","S07","S08","S12"]
SEG_THR   = {"S02":"thr0.08","S03":"thr0.1","S04":"thr0.08","S05":"thr0.25",
             "S06":"thr0.18","S07":"thr0.05","S08":"thr0.18","S12":"thr0.1"}
METL_V1_RUN = {"S02":"XGWfEkoP","S03":"bat4JTEc","S04":"e74qnHH5","S05":"f6ct8j7L",
                "S06":"F7uG4XHT","S07":"WkqJQ2ni","S08":"RCd8d3kN","S12":"N5SHB9oU"}

# ── helpers ──────────────────────────────────────────────────────────────────
def load_sklearn_preds(seg):
    d = BASE / f"esm_results/{seg}"
    with open(d / "mlp_model.pkl", "rb") as f: mlp = pickle.load(f)
    with open(d / "scaler.pkl",    "rb") as f: sc  = pickle.load(f)
    return mlp.predict(sc.transform(np.load(d / "X_test.npy")))

def load_msa_preds(seg):
    d = BASE / f"msa_results/{seg}"
    with open(d / "mlp_model.pkl", "rb") as f: mlp = pickle.load(f)
    with open(d / "scaler.pkl",    "rb") as f: sc  = pickle.load(f)
    return mlp.predict(sc.transform(np.load(d / "X_test.npy")))

def load_metl_preds(run_dir):
    return np.load(run_dir / "predictions/test_predictions.npy").squeeze()

def find_v2_run(seg):
    # Prefer run that has test_predictions.npy (some segs have duplicate runs)
    for d in (BASE / "metl_rosetta/target_model_45k_v2").iterdir():
        hf = d / "hparams.yaml"
        pred = d / "predictions/test_predictions.npy"
        if hf.exists() and f"his3_{seg}_local_45k_v2" in open(hf).read() and pred.exists():
            return d
    return None

def get_test_df(seg, thr):
    df_all  = pd.read_csv(BASE / f"splits_segmentwise_species/super_segments_{seg}_{thr}_all.csv")
    idx     = np.loadtxt(BASE / f"metl_rosetta/metl_splits/{seg}_{thr}/test.txt", dtype=int)
    return df_all.iloc[idx].reset_index(drop=True)

def seg_dist_delta(seg, thr):
    df = pd.read_csv(BASE / f"splits_segmentwise_species/super_segments_{seg}_{thr}_all.csv")
    return (df[df.split=="test"]["nearest_dist_to_ref"].mean()
          - df[df.split=="train"]["nearest_dist_to_ref"].mean())

# ── build segment-level table ────────────────────────────────────────────────
rows = []
for seg in SEGMENTS:
    thr  = SEG_THR[seg]
    delta = seg_dist_delta(seg, thr)
    df_t  = get_test_df(seg, thr)
    y     = df_t["DMS_score"].values

    p_esm  = load_sklearn_preds(seg)
    p_msa  = load_msa_preds(seg)
    p_mv1  = load_metl_preds(BASE / f"metl_rosetta/target_model_45k/{METL_V1_RUN[seg]}")
    v2dir  = find_v2_run(seg)
    p_mv2  = load_metl_preds(v2dir) if v2dir else np.full_like(p_mv1, np.nan)

    rows.append(dict(
        seg   = seg,
        delta = delta,
        esm   = stats.spearmanr(p_esm,  y).statistic,
        msa   = stats.spearmanr(p_msa,  y).statistic,
        mv1   = stats.spearmanr(p_mv1,  y).statistic,
        mv2   = stats.spearmanr(p_mv2,  y).statistic if not np.all(np.isnan(p_mv2)) else np.nan,
        p_esm=p_esm, p_msa=p_msa, p_mv1=p_mv1, p_mv2=p_mv2, y=y,
        dist_test = df_t["nearest_dist_to_ref"].values,
        clusters  = df_t["nearest_species_cluster"].values,
    ))

df_seg = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ("p_esm","p_msa","p_mv1","p_mv2","y",
                                      "dist_test","clusters")} for r in rows])

# ── Figure 1: Performance vs. Δ Distance ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

perf_cols = {"ESM-2 650M":"esm","MSA-T 100M":"msa",
             "METL-Local v1":"mv1","METL-Local v2":"mv2"}

x = df_seg["delta"].values
for label, col in perf_cols.items():
    st = MODEL_STYLE[label]
    y_vals = df_seg[col].values
    ax.scatter(x, y_vals, color=st["color"], marker=st["marker"],
               s=90, zorder=5, label=label)
    # regression line
    valid = ~np.isnan(y_vals)
    if valid.sum() > 2:
        m, b, r, p, _ = stats.linregress(x[valid], y_vals[valid])
        xfit = np.linspace(x.min()-0.01, x.max()+0.01, 100)
        ax.plot(xfit, m*xfit+b, color=st["color"], ls=st["ls"],
                lw=st["lw"], alpha=0.75,
                label=f"  (rho={stats.spearmanr(x[valid], y_vals[valid]).statistic:+.2f})")

# Annotate segments on ESM curve
for _, r in df_seg.iterrows():
    ax.annotate(r.seg, (r.delta, r.esm), textcoords="offset points",
                xytext=(4, 4), fontsize=8, color="#2196F3", alpha=0.85)

ax.set_xlabel("Δ mean phylogenetic distance (test − train)", fontsize=12)
ax.set_ylabel("Spearman ρ on OOD test set", fontsize=12)
ax.set_title("Model Performance vs. Train/Test Phylogenetic Distance\n(per segment)", fontsize=13)
ax.legend(fontsize=9, loc="upper right", ncol=2)
ax.set_ylim(-0.05, 1.0)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "plot_performance_vs_distance.png", dpi=150)
print("Saved plot_performance_vs_distance.png")
plt.close()

# ── Figure 2: Per-cluster analysis for S02 and S08 ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, seg in zip(axes, ["S02", "S08"]):
    row = next(r for r in rows if r["seg"] == seg)
    y, clusters, dists = row["y"], row["clusters"], row["dist_test"]
    p_esm, p_mv1, p_mv2 = row["p_esm"], row["p_mv1"], row["p_mv2"]

    clust_rows = []
    for clust in np.unique(clusters):
        mask = clusters == clust
        if mask.sum() < 8:
            continue
        n = mask.sum()
        d = dists[mask].mean()
        with np.errstate(invalid="ignore"):
            sp_e  = stats.spearmanr(p_esm[mask],  y[mask]).statistic
            sp_m1 = stats.spearmanr(p_mv1[mask],  y[mask]).statistic
            sp_m2 = stats.spearmanr(p_mv2[mask],  y[mask]).statistic
        clust_rows.append(dict(clust=clust, n=n, dist=d,
                               esm=sp_e, mv1=sp_m1, mv2=sp_m2))

    cdf = pd.DataFrame(clust_rows).sort_values("dist")

    # jitter x slightly so overlapping clusters are visible
    jitter = np.random.default_rng(42).uniform(-0.004, 0.004, len(cdf))

    for label, col, st in [("ESM-2 650M","esm",MODEL_STYLE["ESM-2 650M"]),
                             ("METL-Local v1","mv1",MODEL_STYLE["METL-Local v1"]),
                             ("METL-Local v2","mv2",MODEL_STYLE["METL-Local v2"])]:
        sizes = np.clip(cdf["n"].values / cdf["n"].max() * 300, 30, 300)
        ax.scatter(cdf["dist"] + jitter, cdf[col],
                   color=st["color"], marker=st["marker"],
                   s=sizes, alpha=0.8, label=label, zorder=5)
        # regression
        valid = cdf[col].notna() & ~np.isnan(cdf[col])
        if valid.sum() > 2:
            m, b, r, p, _ = stats.linregress(cdf.loc[valid,"dist"], cdf.loc[valid,col])
            xfit = np.linspace(cdf["dist"].min()-0.01, cdf["dist"].max()+0.01, 100)
            rho  = stats.spearmanr(cdf.loc[valid,"dist"], cdf.loc[valid,col]).statistic
            ax.plot(xfit, m*xfit+b, color=st["color"], ls=st["ls"],
                    lw=1.5, alpha=0.6, label=f"  rho={rho:+.2f}")

    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Cluster mean phylogenetic distance to ref", fontsize=11)
    ax.set_ylabel("Spearman ρ within cluster", fontsize=11)
    ax.set_title(f"Per-Cluster Performance — Segment {seg}\n(bubble size ∝ cluster size)", fontsize=12)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.set_ylim(-0.4, 1.05)
    ax.grid(True, alpha=0.3)

fig.suptitle("Does performance drop with phylogenetic distance? (within test set clusters)",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "plot_cluster_distance_analysis.png", dpi=150, bbox_inches="tight")
print("Saved plot_cluster_distance_analysis.png")
plt.close()

# ── Figure 3: ESM vs METL gap per cluster (question 2) ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, seg in zip(axes, ["S02", "S08"]):
    row = next(r for r in rows if r["seg"] == seg)
    y, clusters, dists = row["y"], row["clusters"], row["dist_test"]
    p_esm, p_mv1, p_mv2 = row["p_esm"], row["p_mv1"], row["p_mv2"]

    clust_rows = []
    for clust in np.unique(clusters):
        mask = clusters == clust
        if mask.sum() < 8:
            continue
        d = dists[mask].mean()
        with np.errstate(invalid="ignore"):
            sp_e  = stats.spearmanr(p_esm[mask],  y[mask]).statistic
            sp_m1 = stats.spearmanr(p_mv1[mask],  y[mask]).statistic
            sp_m2 = stats.spearmanr(p_mv2[mask],  y[mask]).statistic
        clust_rows.append(dict(n=mask.sum(), dist=d,
                               gap_v1=sp_e-sp_m1, gap_v2=sp_e-sp_m2))

    cdf = pd.DataFrame(clust_rows).sort_values("dist")
    sizes = np.clip(cdf["n"].values / cdf["n"].max() * 300, 30, 300)

    ax.scatter(cdf["dist"], cdf["gap_v1"], s=sizes,
               color=MODEL_STYLE["METL-Local v1"]["color"],
               marker="^", alpha=0.8, label="ESM − METL-v1 gap", zorder=5)
    ax.scatter(cdf["dist"], cdf["gap_v2"], s=sizes,
               color=MODEL_STYLE["METL-Local v2"]["color"],
               marker="D", alpha=0.8, label="ESM − METL-v2 gap", zorder=5)

    for col, clr in [("gap_v1", MODEL_STYLE["METL-Local v1"]["color"]),
                      ("gap_v2", MODEL_STYLE["METL-Local v2"]["color"])]:
        valid = cdf[col].notna()
        if valid.sum() > 2:
            m, b, r, p, _ = stats.linregress(cdf.loc[valid,"dist"], cdf.loc[valid,col])
            xfit = np.linspace(cdf["dist"].min()-0.01, cdf["dist"].max()+0.01, 100)
            rho  = stats.spearmanr(cdf.loc[valid,"dist"], cdf.loc[valid,col]).statistic
            ax.plot(xfit, m*xfit+b, color=clr, lw=2, alpha=0.7,
                    label=f"  rho={rho:+.2f}")

    ax.axhline(0, color="gray", lw=1, ls="--", label="no gap")
    ax.set_xlabel("Cluster mean phylogenetic distance", fontsize=11)
    ax.set_ylabel("ESM − METL Spearman gap\n(positive = ESM better)", fontsize=11)
    ax.set_title(f"Where does METL fall behind ESM? — {seg}\n(bubble ∝ cluster size)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("METL-Local vs ESM performance gap per cluster\n"
             "If gap grows with distance → METL relies on species-specific features",
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "plot_metl_esm_gap.png", dpi=150, bbox_inches="tight")
print("Saved plot_metl_esm_gap.png")
plt.close()

print("\nDone. All plots saved to", OUT)
