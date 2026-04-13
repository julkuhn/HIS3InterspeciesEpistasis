#!/usr/bin/env python3
"""
Erstellt SQLite-Datenbank aus Energize-Outputs für METL Source Model Training.

Schritte:
  1. Alle energies.csv aus energize_outputs/ einlesen
  2. Daten bereinigen (NaN, Duplikate entfernen)
  3. SQLite-Datenbank erstellen
  4. pdb_fns.txt erstellen
  5. Train/Val/Test-Splits erstellen (80/10/10)
  6. Standardisierungsparameter berechnen
"""

import os
import sys
import glob
import sqlite3
import argparse
import logging
from os.path import join, dirname, isfile, isdir

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ENERGY_COLS_START = "total_score"  # first energy column in the CSV


def find_energies_files(energize_outputs_dir):
    pattern = join(energize_outputs_dir, "**", "energies.csv")
    files = glob.glob(pattern, recursive=True)
    log.info(f"Gefunden: {len(files)} energies.csv Dateien in {energize_outputs_dir}")
    return sorted(files)


def load_and_clean(energies_files):
    dfs = []
    skipped = 0
    for fn in energies_files:
        try:
            df = pd.read_csv(fn)
            dfs.append(df)
        except Exception as e:
            log.warning(f"Überspringe {fn}: {e}")
            skipped += 1
    if skipped > 0:
        log.warning(f"{skipped} Dateien übersprungen")

    log.info("Concateniere DataFrames...")
    rosetta = pd.concat(dfs, ignore_index=True)
    log.info(f"  Gesamt: {len(rosetta)} Varianten")

    # Spalte umbenennen: variant -> mutations (METL-Konvention)
    if "variant" in rosetta.columns and "mutations" not in rosetta.columns:
        rosetta = rosetta.rename(columns={"variant": "mutations"})

    # Duplikate entfernen
    before = len(rosetta)
    rosetta = rosetta.drop_duplicates(subset=["pdb_fn", "mutations"])
    log.info(f"  Nach Duplikat-Entfernung: {len(rosetta)} ({before - len(rosetta)} entfernt)")

    # NaN in total_score entfernen
    before = len(rosetta)
    rosetta = rosetta.dropna(subset=["total_score"])
    log.info(f"  Nach NaN-Entfernung: {len(rosetta)} ({before - len(rosetta)} entfernt)")

    log.info(f"Statistiken total_score: "
             f"mean={rosetta['total_score'].mean():.2f}, "
             f"std={rosetta['total_score'].std():.2f}, "
             f"range=[{rosetta['total_score'].min():.2f}, {rosetta['total_score'].max():.2f}]")

    return rosetta


def create_sqlite_db(df, db_fn, ct_fn):
    """Erstellt SQLite-Datenbank mit dem korrekten Schema und füllt sie."""
    if isfile(db_fn):
        log.warning(f"DB existiert bereits, wird überschrieben: {db_fn}")
        os.remove(db_fn)

    os.makedirs(dirname(db_fn), exist_ok=True)

    con = sqlite3.connect(db_fn)

    # Schema aus SQL-Datei laden
    with open(ct_fn, "r") as f:
        sql = f.read()
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            con.execute(stmt)
    con.commit()

    # pdb_file Tabelle befüllen
    for pdb_fn in df["pdb_fn"].unique():
        con.execute(
            "INSERT OR IGNORE INTO pdb_file (pdb_fn) VALUES (?)",
            (pdb_fn,)
        )
    con.commit()

    # job Tabelle befüllen (falls job_uuid vorhanden)
    if "job_uuid" in df.columns:
        for uuid in df["job_uuid"].unique():
            con.execute(
                "INSERT OR IGNORE INTO job (uuid, cluster, process) VALUES (?, ?, ?)",
                (uuid, "local", "local")
            )
        con.commit()

    # variant Tabelle befüllen
    log.info(f"Schreibe {len(df)} Varianten in DB...")

    # Spalten die im Schema existieren
    schema_cols = [
        "pdb_fn", "mutations", "job_uuid",
        "start_time", "run_time", "mutate_run_time", "relax_run_time",
        "filter_run_time", "centroid_run_time",
        "total_score", "dslf_fa13", "fa_atr", "fa_dun", "fa_elec",
        "fa_intra_rep", "fa_intra_sol_xover4", "fa_rep", "fa_sol",
        "hbond_bb_sc", "hbond_lr_bb", "hbond_sc", "hbond_sr_bb",
        "lk_ball_wtd", "omega", "p_aa_pp", "pro_close", "rama_prepro",
        "ref", "yhh_planarity", "filter_total_score",
        "buried_all", "buried_np", "contact_all", "contact_buried_core",
        "contact_buried_core_boundary", "degree", "degree_core",
        "degree_core_boundary", "exposed_hydrophobics", "exposed_np_AFIMLWVY",
        "exposed_polars", "exposed_total", "one_core_each", "pack",
        "res_count_all", "res_count_buried_core", "res_count_buried_core_boundary",
        "res_count_buried_np_core", "res_count_buried_np_core_boundary",
        "ss_contributes_core", "ss_mis", "total_hydrophobic",
        "total_hydrophobic_AFILMVWY", "total_sasa", "two_core_each", "unsat_hbond",
        "centroid_total_score", "cbeta", "cenpack", "env", "hs_pair",
        "linear_chainbreak", "overlap_chainbreak", "pair", "rg", "rsigma",
        "sheet", "ss_pair", "vdw",
    ]

    # Nur Spalten nehmen die auch im DataFrame vorhanden sind
    available_cols = [c for c in schema_cols if c in df.columns]

    # Placeholder für fehlende Pflicht-Spalten
    df_out = df[available_cols].copy()
    if "job_uuid" not in df_out.columns:
        df_out["job_uuid"] = "unknown"

    placeholders = ", ".join(["?"] * len(available_cols))
    col_str = ", ".join([f"`{c}`" for c in available_cols])
    insert_sql = f"INSERT OR IGNORE INTO variant ({col_str}) VALUES ({placeholders})"

    chunk_size = 10000
    rows = [tuple(row) for row in df_out.itertuples(index=False, name=None)]
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        con.executemany(insert_sql, chunk)
        con.commit()
        log.info(f"  Eingefügt: {min(i + chunk_size, len(rows))}/{len(rows)}")

    # Rowcount prüfen
    count = con.execute("SELECT COUNT(*) FROM variant").fetchone()[0]
    log.info(f"DB enthält {count} Varianten")

    con.close()
    return count


def create_pdb_fns_txt(df, out_dir):
    """Erstellt pdb_fns.txt neben der Datenbank."""
    pdb_fns = df["pdb_fn"].tolist()  # alle Zeilen (mit Duplikaten für row->pdb mapping)
    fn = join(out_dir, "pdb_fns.txt")
    pd.Series(pdb_fns).to_csv(fn, index=False, header=False)
    log.info(f"pdb_fns.txt erstellt: {len(pdb_fns)} Einträge, {df['pdb_fn'].nunique()} unique PDBs")


def create_splits(n_total, split_dir, train_size=0.8, val_size=0.1, seed=42):
    """Erstellt Train/Val/Test-Split-Dateien (Indizes)."""
    os.makedirs(split_dir, exist_ok=True)

    indices = np.arange(n_total)
    test_size = 1.0 - train_size - val_size

    train_idx, temp_idx = train_test_split(indices, train_size=train_size, random_state=seed)
    val_ratio_of_temp = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(temp_idx, train_size=val_ratio_of_temp, random_state=seed)

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        fn = join(split_dir, f"{name}.txt")
        np.savetxt(fn, idx, fmt="%d")
        log.info(f"  {name}: {len(idx)} Varianten → {fn}")

    return train_idx, val_idx, test_idx


def compute_standardization(df, train_idx, split_dir, energy_start_col="total_score"):
    """Berechnet Mittelwert und Standardabweichung der Energy-Features (nur Train-Set)."""
    params_dir = join(split_dir, "standardization_params")
    os.makedirs(params_dir, exist_ok=True)

    # Nur Train-Set für Standardisierung verwenden
    train_df = df.iloc[train_idx]

    # Alle numerischen Spalten ab energy_start_col
    col_names = list(df.columns)
    start_idx = col_names.index(energy_start_col)
    energy_cols = col_names[start_idx:]
    energy_cols = [c for c in energy_cols if pd.api.types.is_numeric_dtype(df[c])]

    # Berechnung pro PDB (grouped)
    g = train_df.groupby("pdb_fn")[energy_cols]
    g_mean = g.mean()
    g_std = g.std(ddof=0)  # biased estimator (wie sklearn StandardScaler)

    means_fn = join(params_dir, "energy_means_train.tsv")
    stds_fn = join(params_dir, "energy_stds_train.tsv")

    g_mean.to_csv(means_fn, sep="\t", float_format="%.7f")
    g_std.to_csv(stds_fn, sep="\t", float_format="%.7f")

    log.info(f"Standardisierungsparameter gespeichert in {params_dir}")
    log.info(f"  {len(energy_cols)} Energy-Features, {train_df['pdb_fn'].nunique()} PDBs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energize_outputs_dir", required=True,
                        help="Verzeichnis mit allen energize chunk-Outputs")
    parser.add_argument("--db_fn", required=True,
                        help="Pfad zur Ausgabe-SQLite-Datenbank")
    parser.add_argument("--split_dir", required=True,
                        help="Verzeichnis für Train/Val/Test-Splits")
    parser.add_argument("--ct_fn",
                        default="/workspace/metl-sim/variant_database/create_tables.sql",
                        help="Pfad zur SQL create_tables.sql Datei")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Dateien finden und laden
    energies_files = find_energies_files(args.energize_outputs_dir)
    if not energies_files:
        log.error("Keine energies.csv Dateien gefunden!")
        sys.exit(1)

    df = load_and_clean(energies_files)

    # 2. SQLite-Datenbank erstellen
    db_dir = dirname(args.db_fn)
    create_sqlite_db(df, args.db_fn, args.ct_fn)

    # 3. pdb_fns.txt erstellen (eine Zeile pro Variante = row-Reihenfolge)
    create_pdb_fns_txt(df, db_dir)

    # 4. Splits erstellen
    log.info(f"Erstelle Splits ({args.train_size}/{args.val_size}/{1-args.train_size-args.val_size:.1f})...")
    train_idx, val_idx, test_idx = create_splits(
        n_total=len(df),
        split_dir=args.split_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    # 5. Standardisierungsparameter berechnen
    log.info("Berechne Standardisierungsparameter...")
    compute_standardization(df, train_idx, args.split_dir)

    log.info("Fertig!")
    log.info(f"  DB:        {args.db_fn}")
    log.info(f"  Splits:    {args.split_dir}")
    log.info(f"  Varianten: {len(df)}")


if __name__ == "__main__":
    main()
