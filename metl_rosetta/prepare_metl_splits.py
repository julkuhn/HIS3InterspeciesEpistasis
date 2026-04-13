#!/usr/bin/env python3
"""Prepare METL dataset TSV files and split directories for all segments."""

import os
import pandas as pd
import yaml

FOPRA    = "/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge28jel2/fopra"
SPLITS_DIR   = f"{FOPRA}/splits_segmentwise_species"
OUT_SPLITS   = f"{FOPRA}/metl_rosetta/metl_splits"
DATASETS_YML = f"{FOPRA}/metl/data/dms_data/datasets.yml"

WT_AA = ("MTEQKALVKRITNETKIQIAISLKGGPLAIEHSIFPEKEAEAVAEQATQSQVINVHTGIGFLDHMIHALA"
         "KHSGWSLIVECIGDLHIDDHHTTEDCGIALGQAFKEALGAVRGVKRFGSGFAPLDEALSRAVVDLSNRPY"
         "AVVELGLQREKVGDLSCEMIPHFLESFAEASRITLHVDCLRGKNDHHRSESAFKALAVAIREATSPNGTND"
         "VPSTKGVLM")

best = pd.read_csv(f"{SPLITS_DIR}/best_per_segment.csv")

with open(DATASETS_YML) as f:
    datasets = yaml.safe_load(f)

os.makedirs(OUT_SPLITS, exist_ok=True)

for _, row in best.iterrows():
    seg = row["segment_id"]
    thr = row["threshold"]
    ds_name  = f"his3_{seg}_thr{thr}"
    csv_fn   = f"{SPLITS_DIR}/super_segments_{seg}_thr{thr}_all.csv"
    tsv_fn   = f"{SPLITS_DIR}/super_segments_{seg}_thr{thr}_all.tsv"
    split_dir = f"{OUT_SPLITS}/{seg}_thr{thr}"

    if not os.path.exists(csv_fn):
        print(f"WARNING: missing {csv_fn}, skipping")
        continue

    print(f"\n{seg} (thr={thr})")

    df = pd.read_csv(csv_fn)

    if not os.path.exists(tsv_fn):
        df_out = df.rename(columns={"mutant": "variant"})
        df_out["variant"] = df_out["variant"].str.replace(":", ",")
        df_out.to_csv(tsv_fn, sep="\t", index=False)
        print(f"  wrote TSV ({len(df)} rows)")
    else:
        print(f"  TSV already exists")

    os.makedirs(split_dir, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        idxs = df.index[df["split"] == split_name].tolist()
        with open(f"{split_dir}/{split_name}.txt", "w") as f:
            f.write("\n".join(map(str, idxs)) + "\n")
        print(f"  {split_name}: {len(idxs)}")

    if ds_name not in datasets:
        datasets[ds_name] = {
            "ds_fn":     f"/workspace/splits_segmentwise_species/super_segments_{seg}_thr{thr}_all.tsv",
            "wt_aa":     WT_AA,
            "wt_ofs":    0,
            "encoding":  "int_seqs",
            "target_names": ["DMS_score"],
        }
        print(f"  added '{ds_name}' to datasets.yml")
    else:
        print(f"  '{ds_name}' already in datasets.yml")

with open(DATASETS_YML, "w") as f:
    yaml.dump(datasets, f, default_flow_style=False, allow_unicode=True)

print("\nDone.")
