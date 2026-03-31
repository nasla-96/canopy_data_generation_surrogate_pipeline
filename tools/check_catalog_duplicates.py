#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse

ID_COL = "simulation_id"


def check_duplicates(csv_path, tol=1e-12):
    df = pd.read_csv(csv_path)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # -----------------------------
    # 1. Check duplicate simulation_id
    # -----------------------------
    dup_ids = df[df[ID_COL].duplicated()]

    print("\n=== DUPLICATE simulation_id ===")
    if len(dup_ids) == 0:
        print("No duplicate simulation_id values.")
    else:
        print("Found duplicates:", len(dup_ids))
        print(dup_ids.head())

    # -----------------------------
    # 2. Check exact duplicate samples
    # -----------------------------
    param_cols = [c for c in df.columns if c != ID_COL]

    dup_samples = df[df[param_cols].duplicated()]

    print("\n=== EXACT DUPLICATE PARAMETER ROWS ===")
    if len(dup_samples) == 0:
        print("No exact duplicate Sobol samples.")
    else:
        print("Found duplicates:", len(dup_samples))
        print(dup_samples[[ID_COL]].head())

    # -----------------------------
    # 3. Check duplicates ignoring tiny float differences
    # -----------------------------
    print("\n=== NEAR DUPLICATES (tolerance check) ===")

    rounded = df[param_cols].round(int(-np.log10(tol)))

    dup_near = rounded[rounded.duplicated()]

    if len(dup_near) == 0:
        print("No near-duplicate samples.")
    else:
        print("Near duplicates found:", len(dup_near))

    # -----------------------------
    # 4. Print duplicate groups
    # -----------------------------
    if len(dup_samples) > 0:
        print("\nExample duplicate groups:")

        groups = (
            df.groupby(param_cols)[ID_COL]
            .apply(list)
            .reset_index(name="ids")
        )

        groups = groups[groups["ids"].apply(len) > 1]

        print(groups.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)

    args = parser.parse_args()

    check_duplicates(args.csv)