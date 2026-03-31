#!/usr/bin/env python3
"""
scripts/build_ml_dataset.py

Merge:
  - scaled catalog CSV (inputs)
  - Helios results CSV (contains net_PAR)

Output:
  - ML-ready CSV with ONLY parameters + net_PAR (and simulation_id)
"""

import os
import argparse
import pandas as pd

DROP_COLS_DEFAULT = [
    "status", "error", "stage", "obj_path", "config_path", "device_id"
]

def main(scaled_catalog_csv: str, results_csv: str, out_csv: str) -> None:
    X = pd.read_csv(scaled_catalog_csv)
    y = pd.read_csv(results_csv)

    # Keep only successful rows with finite target
    if "status" in y.columns:
        y = y.loc[y["status"] == "SUCCESS"].copy()

    if "net_PAR" not in y.columns:
        raise ValueError("results_csv must contain net_PAR")

    # Merge
    df = X.merge(y[["simulation_id", "net_PAR"] + [c for c in DROP_COLS_DEFAULT if c in y.columns]],
                 on="simulation_id", how="inner")

    # Drop metadata columns if present
    for c in DROP_COLS_DEFAULT:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Drop any NaN targets
    df = df[pd.notnull(df["net_PAR"])].copy()

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote ML dataset CSV: {out_csv}  (rows={len(df)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaled_catalog", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.scaled_catalog, args.results, args.out)
