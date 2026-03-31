#!/usr/bin/env python3
# scripts/generate_catalog.py
#
# Generate (or extend) a Sobol catalog for the Helios surrogate pipeline.
# OUTPUT IS CSV (not parquet) per your updated workflow.
#
# Notes on extending:
# - If you already have an existing catalog CSV, use --append to add new sims.
# - New rows will be assigned simulation_id starting at (max_id + 1).
# - Sobol is generated with scramble=True; if you want repeatability for *new* batches,
#   pass an explicit --seed. (Your original 2500-run catalog may not be reproducible
#   if it was generated without a seed.)

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import qmc
from typing import Optional

NUM_LEAVES = 12
D_GEOM = 1 + 7 * NUM_LEAVES        # 85
D_TOTAL = D_GEOM + 2               # + lat, lon = 87

# Fixed longitude: Ames, IA (default)
AMES_LON_DEG = 93.62


def generate_sobol_points(n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)

    # Sobol performs best with power-of-2 samples; generate enough and truncate.
    m = int(np.ceil(np.log2(max(1, n))))
    pts = sampler.random_base2(m=m)  # shape: (2**m, d)
    return pts[:n, :]


def map_to_range(x, lo, hi):
    return lo + x * (hi - lo)


def build_catalog_df(sobol: np.ndarray, start_sim_id: int, lon_deg_fixed: float) -> pd.DataFrame:
    n_sims = sobol.shape[0]

    sobol_geom = sobol[:, :D_GEOM]
    sobol_lat  = sobol[:, D_GEOM + 0]

    # Latitude varies; longitude fixed
    lat_deg = map_to_range(sobol_lat, 7.0, 48.0)
    lon_deg = np.full(n_sims, lon_deg_fixed, dtype=float)

    records = []
    for k in range(n_sims):
        sim_id = start_sim_id + k
        g = sobol_geom[k, :]

        row = {
            "simulation_id": int(sim_id),
            "lat_deg": float(lat_deg[k]),
            "lon_deg": float(lon_deg[k]),
        }

        row["stalk_raw"] = float(g[0])

        idx = 1
        for i in range(NUM_LEAVES):
            row[f"interleaf_raw_{i}"] = float(g[idx]); idx += 1
        for i in range(NUM_LEAVES):
            row[f"length_raw_{i}"]    = float(g[idx]); idx += 1
        for i in range(NUM_LEAVES):
            row[f"width_raw_{i}"]     = float(g[idx]); idx += 1
        for i in range(NUM_LEAVES):
            row[f"theta_raw_{i}"]     = float(g[idx]); idx += 1
        for i in range(NUM_LEAVES):
            row[f"phi_raw_{i}"]       = float(g[idx]); idx += 1
        for i in range(NUM_LEAVES):
            row[f"curv_raw_{i}"]      = float(g[idx]); idx += 1
        for i in range(NUM_LEAVES):
            row[f"twist_raw_{i}"]     = float(g[idx]); idx += 1

        records.append(row)

    return pd.DataFrame.from_records(records)


def main(n_sims: int, out_path: str, lon_deg_fixed: float = AMES_LON_DEG,
         append: bool = False, seed: Optional[int] = None) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if append and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        prev = pd.read_csv(out_path)
        start_sim_id = int(prev["simulation_id"].max()) + 1
    else:
        prev = None
        start_sim_id = 0

    sobol = generate_sobol_points(n_sims, D_TOTAL, seed=seed)
    df_new = build_catalog_df(sobol, start_sim_id=start_sim_id, lon_deg_fixed=lon_deg_fixed)

    if prev is not None:
        df = pd.concat([prev, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(out_path, index=False)

    if append and prev is not None:
        print(f"Appended {n_sims} sims to {out_path} (new ids: {start_sim_id}..{start_sim_id + n_sims - 1})")
    else:
        print(f"Saved catalog with {len(df)} sims to {out_path}")

    print(f"Longitude fixed at {lon_deg_fixed}° (Ames default)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or extend a Sobol catalog (CSV) for Helios surrogate pipeline.")
    parser.add_argument("--n_sims", type=int, required=True, help="Number of NEW sims to generate.")
    parser.add_argument("--out_path", type=str, default="catalog/sim_catalog.csv")
    parser.add_argument("--lon_deg", type=float, default=AMES_LON_DEG)
    parser.add_argument("--append", action="store_true", help="Append new sims to an existing catalog CSV.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for scrambled Sobol generator.")

    args = parser.parse_args()
    main(n_sims=args.n_sims, out_path=args.out_path, lon_deg_fixed=args.lon_deg, append=args.append, seed=args.seed)
