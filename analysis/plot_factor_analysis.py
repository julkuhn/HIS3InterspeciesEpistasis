#!/usr/bin/env python3
"""
What factors drive model performance?

Panel A — Segment-level: correlate performance with
  frac_func, score_std, score_range, n_train, n_mut_mean

Panel B — Per-cluster (S02+S08): correlate cluster Spearman with
  cluster size, score_std, frac_func, n_mut_mean

Panel C — Per-sequence residual analysis:
  |prediction error| vs n_mut, functional/non-functional
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

def load_esm(seg):
    d = BASE / f"esm_results/{seg}"
    with open(d/"mlp_model.pkl","rb") as f: m = pickle.load(f)
    with open(d/"scaler.pkl","rb") as f:    s = pickle.load(f)
    return m.predict(s.transform(np.load(d/"X_test.npy")))

def load_metl_v1(seg):
    return np.load(BASE/f"metl_rosetta/target_model_45k/{METL_V1[seg]}/predictions/test_predictions.npy").squeeze()

def find_v2(seg):
    for d in (BASE/"metl_rosetta/target_model_45k_v2").iterdir():
        hf = d/"hparams.yaml"; pred = d/"predictions/test_predictions.npy"
        if hf.exists() and f"his3_{seg}_local_45k_v2" in open(hf).read() and pred.exists():
            return np.load(pred).squeeze()
    return None

def get_test(seg, thr):
    df = pd.read_csv(BASE/f"splits_segmentwise_species/super_segments_{seg}_{thr}_all.csv")
    idx = np.loadtxt(BASE/f"metl_rosetta/metl_splits/{seg}_{thr}/test.txt", dtype=int)
    return df.iloc[idx].reset_index(drop=True)

# ── collect segment-level data ───────────────────────────────────────────────
seg_rows = []
for seg, thr in SEG_THR.items():
    df_all = pd.read_csv(BASE/f"splits_segmentwise_species/super_segments_{seg}_{thr}_all.csv")
    df_te  = get_test(seg, thr)
    y      = df_te["DMS_score"].values
    p_esm  = load_esm(seg)
    p_mv1  = load_metl_v1(seg)
    p_mv2  = find_v2(seg)

    seg_rows.append(dict(
        seg        = seg,
        n_train    = (df_all.split=="train").sum(),
        n_test     = len(df_te),
        frac_func  = df_te["DMS_score_bin"].mean(),
        score_std  = y.std(),
        score_range= y.max()-y.min(),
        n_mut_mean = df_te["n_mut"].mean(),
        dist_delta = df_all[df_all.split=="test"]["nearest_dist_to_ref"].mean()
                   - df_all[df_all.split=="train"]["nearest_dist_to_ref"].mean(),
        sp_esm     = stats.spearmanr(p_esm, y).statistic,
        sp_mv1     = stats.spearmanr(p_mv1, y).statistic,
        sp_mv2     = stats.spearmanr(p_mv2, y).statistic if p_mv2 is not None else np.nan,
        p_esm=p_esm, p_mv1=p_mv1, p_mv2=p_mv2, y=y, df_te=df_te,
    ))

sdf = pd.DataFrame([{k:v for k,v in r.items()
                      if k not in ("p_esm","p_mv1","p_mv2","y","df_te")} for r in seg_rows])

# ── Figure A: Segment-level factor correlations ──────────────────────────────
factors = [
    ("frac_func",   "Fraction functional\n(DMS_score_bin=1)"),
    ("score_std",   "DMS score std dev\n(label spread)"),
    ("score_range", "DMS score range\n(max − min)"),
    ("n_train",     "Training set size"),
    ("n_mut_mean",  "Mean #mutations\nin test sequences"),
    ("dist_delta",  "Δ phylogenetic distance\n(test − train)"),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for ax, (fcol, flabel) in zip(axes, factors):
    x = sdf[fcol].values
    for model, col, c in [("ESM","sp_esm",COLORS["ESM"]),
                           ("METL-v1","sp_mv1",COLORS["METL-v1"]),
                           ("METL-v2","sp_mv2",COLORS["METL-v2"])]:
        y_sp = sdf[col].values
        valid = ~np.isnan(y_sp)
        rho, p = stats.spearmanr(x[valid], y_sp[valid])
        ax.scatter(x[valid], y_sp[valid], label=f"{model} (ρ={rho:+.2f}, p={p:.2f})",
                   color=c, s=80, zorder=5)
        if valid.sum() > 2:
            m, b = np.polyfit(x[valid], y_sp[valid], 1)
            xf = np.linspace(x.min(), x.max(), 100)
            ax.plot(xf, m*xf+b, color=c, lw=1.5, alpha=0.6)

    # Annotate with segment names (ESM only for clarity)
    for _, row in sdf.iterrows():
        ax.annotate(row.seg, (row[fcol], row.sp_esm),
                    xytext=(3,3), textcoords="offset points", fontsize=7.5, color=COLORS["ESM"])

    ax.set_xlabel(flabel, fontsize=10)
    ax.set_ylabel("Spearman ρ (OOD test)", fontsize=10)
    ax.legend(fontsize=7.5, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.0)

fig.suptitle("What drives segment-level performance?\n(each dot = one segment, n=8)",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT/"plot_factors_segment_level.png", dpi=150, bbox_inches="tight")
print("Saved plot_factors_segment_level.png")
plt.close()

# ── Figure B: Per-cluster factor analysis (S02 + S08) ───────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 10))

for row_i, seg in enumerate(["S02", "S08"]):
    r    = next(x for x in seg_rows if x["seg"]==seg)
    df_t = r["df_te"]
    y    = r["y"]
    clusters = df_t["nearest_species_cluster"].values

    clust_rows = []
    for clust in np.unique(clusters):
        mask = clusters == clust
        if mask.sum() < 8: continue
        ct = df_t[mask]
        sp_e  = stats.spearmanr(r["p_esm"][mask],  y[mask]).statistic
        sp_m1 = stats.spearmanr(r["p_mv1"][mask],  y[mask]).statistic
        sp_m2 = stats.spearmanr(r["p_mv2"][mask],  y[mask]).statistic if r["p_mv2"] is not None else np.nan
        clust_rows.append(dict(
            n          = mask.sum(),
            score_std  = y[mask].std(),
            score_range= y[mask].max()-y[mask].min(),
            frac_func  = ct["DMS_score_bin"].mean(),
            n_mut_mean = ct["n_mut"].mean(),
            dist       = ct["nearest_dist_to_ref"].mean(),
            sp_esm=sp_e, sp_mv1=sp_m1, sp_mv2=sp_m2,
        ))

    cdf = pd.DataFrame(clust_rows)

    cfactors = [
        ("n",           "Cluster size (n)"),
        ("score_std",   "Score std in cluster"),
        ("frac_func",   "Fraction functional"),
        ("n_mut_mean",  "Mean #mutations"),
    ]

    for col_i, (fcol, flabel) in enumerate(cfactors):
        ax = axes[row_i][col_i]
        x  = cdf[fcol].values
        for model, col, c, mk in [("ESM","sp_esm",COLORS["ESM"],"o"),
                                    ("METL-v1","sp_mv1",COLORS["METL-v1"],"^"),
                                    ("METL-v2","sp_mv2",COLORS["METL-v2"],"D")]:
            yv = cdf[col].values
            valid = ~np.isnan(yv)
            rho, p = stats.spearmanr(x[valid], yv[valid])
            sz = np.clip(cdf["n"].values/cdf["n"].max()*200, 20, 200)
            ax.scatter(x[valid], yv[valid], label=f"{model} ρ={rho:+.2f}",
                       color=c, marker=mk, s=sz[valid], alpha=0.8, zorder=5)
            if valid.sum() > 3:
                m, b = np.polyfit(x[valid], yv[valid], 1)
                xf = np.linspace(x.min(), x.max(), 100)
                ax.plot(xf, m*xf+b, color=c, lw=1.5, alpha=0.55)

        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.set_xlabel(flabel, fontsize=9)
        ax.set_ylabel(f"Spearman ρ within cluster\n({seg})" if col_i==0 else "", fontsize=9)
        ax.set_title(f"{seg}: {flabel}", fontsize=9.5)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.45, 1.05)

fig.suptitle("Per-cluster: Which factors explain within-cluster performance?\n(bubble size ∝ cluster size)",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT/"plot_factors_cluster_level.png", dpi=150, bbox_inches="tight")
print("Saved plot_factors_cluster_level.png")
plt.close()

# ── Figure C: Per-sequence residual vs n_mut and functional ─────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for row_i, seg in enumerate(["S02", "S08"]):
    r    = next(x for x in seg_rows if x["seg"]==seg)
    df_t = r["df_te"]
    y    = r["y"]
    func = df_t["DMS_score_bin"].values.astype(bool)
    n_mut = df_t["n_mut"].values

    for col_i, (model, pred, c) in enumerate([
        ("ESM",     r["p_esm"], COLORS["ESM"]),
        ("METL-v1", r["p_mv1"], COLORS["METL-v1"]),
        ("METL-v2", r["p_mv2"], COLORS["METL-v2"]),
    ]):
        ax = axes[row_i][col_i]
        if pred is None:
            ax.text(0.5,0.5,"N/A",ha="center",va="center",transform=ax.transAxes)
            continue

        err = np.abs(pred - y)  # absolute error (after standardizing)

        # Box plot: error by n_mut
        n_vals = sorted(df_t["n_mut"].unique())
        bp_data = [err[n_mut==n] for n in n_vals]
        bp = ax.boxplot(bp_data, positions=n_vals, widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(c); patch.set_alpha(0.5)

        # Overlay: functional vs non-functional median per n_mut
        for n in n_vals:
            mask_n = n_mut == n
            for is_func, mk, lbl in [(True,"^","func"),(False,"v","non-func")]:
                m = mask_n & (func == is_func)
                if m.sum() >= 3:
                    ax.scatter(n, np.median(err[m]), color=c if is_func else "gray",
                               marker=mk, s=60, zorder=6, alpha=0.85)

        # Spearman: n_mut vs error
        rho, pval = stats.spearmanr(n_mut, err)
        # Mann-Whitney: func vs non-func error
        if func.sum() > 5 and (~func).sum() > 5:
            stat, mw_p = stats.mannwhitneyu(err[func], err[~func], alternative="two-sided")
        else:
            mw_p = float("nan")

        ax.set_xlabel("Number of mutations", fontsize=10)
        ax.set_ylabel("|Prediction error|" if col_i==0 else "", fontsize=10)
        ax.set_title(f"{seg} — {model}\nn_mut ρ={rho:+.2f} p={pval:.3f} | func vs non-func p={mw_p:.3f}",
                     fontsize=9.5)
        ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("Per-sequence absolute error: effect of #mutations and functional status\n"
             "(▲=functional median, ▼=non-functional median per #mut level)",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT/"plot_factors_per_sequence.png", dpi=150, bbox_inches="tight")
print("Saved plot_factors_per_sequence.png")
plt.close()

# ── Summary table ────────────────────────────────────────────────────────────
print("\n=== Segment-level factor correlations with ESM Spearman ===")
x_esc = sdf["sp_esm"].values
for fcol, flabel in factors:
    rho, p = stats.spearmanr(sdf[fcol].values, x_esc)
    print(f"  {fcol:<15}: rho={rho:+.3f}  p={p:.3f}")

print("\nDone.")
