#!/usr/bin/env python3
# scripts/generate_canopy.py
#
# UPDATED: catalog can be CSV or Parquet (auto-detected by extension),
# defaulting to CSV per your new pipeline.


import os
import sys
import numpy as np
import pandas as pd
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from maize_canopy_pipeline.geometry.canopy_from_sobol import generate_canopy_obj_from_sobol

NUM_LEAVES = 12
D_GEOM = 1 + 7 * NUM_LEAVES  # 85
CATALOG_PATH = "catalog/sim_catalog.csv"


def load_catalog(catalog_path: str) -> pd.DataFrame:
    ext = os.path.splitext(catalog_path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(catalog_path)
    return pd.read_csv(catalog_path)


def row_to_sobol_geom(row) -> np.ndarray:
    """Pack the 85 geometry raw dims from a catalog row into a Sobol vector."""
    vec = np.zeros(D_GEOM, dtype=float)
    idx = 0

    vec[idx] = row["stalk_raw"]; idx += 1

    for i in range(NUM_LEAVES):
        vec[idx] = row[f"interleaf_raw_{i}"]; idx += 1
    for i in range(NUM_LEAVES):
        vec[idx] = row[f"length_raw_{i}"];    idx += 1
    for i in range(NUM_LEAVES):
        vec[idx] = row[f"width_raw_{i}"];     idx += 1
    for i in range(NUM_LEAVES):
        vec[idx] = row[f"theta_raw_{i}"];     idx += 1
    for i in range(NUM_LEAVES):
        vec[idx] = row[f"phi_raw_{i}"];       idx += 1
    for i in range(NUM_LEAVES):
        vec[idx] = row[f"curv_raw_{i}"];      idx += 1
    for i in range(NUM_LEAVES):
        vec[idx] = row[f"twist_raw_{i}"];     idx += 1

    return vec


def generate_one_canopy(
    sim_id: int,
    base_json_path: str,
    out_dir: str,
    df: pd.DataFrame,
    row_spacing: float = 0.76,
    plant_spacing: float = 0.20,
    rotation_csv_path: str = None,
    skip_existing: bool = False,
) -> dict:
    """Single-sim tool: catalog row → Sobol vec → plant+field generator."""
    expected_out = os.path.join(out_dir, f"sim_{sim_id:06d}_field_cropped.obj")
    if skip_existing and os.path.exists(expected_out) and os.path.getsize(expected_out) > 0:
        return {"simulation_id": sim_id, "status": "SKIPPED", "obj_path": expected_out}

    row = df.loc[df["simulation_id"] == sim_id]
    if row.empty:
        return {"simulation_id": sim_id, "status": "FAILED", "error": "sim_id_not_in_catalog"}

    row = row.iloc[0]
    sobol_vec = row_to_sobol_geom(row)

    try:
        out_path = generate_canopy_obj_from_sobol(
            sim_id=sim_id,
            sobol_vec=sobol_vec,
            base_json_path=base_json_path,
            out_dir=out_dir,
            row_spacing=row_spacing,
            plant_spacing=plant_spacing,
            rotation_csv_path=rotation_csv_path,
        )
    except Exception as e:
        return {"simulation_id": sim_id, "status": "FAILED", "error": f"geom_exception: {e}"}

    if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        return {"simulation_id": sim_id, "status": "FAILED", "error": "obj_missing_or_empty"}

    return {"simulation_id": sim_id, "status": "SUCCESS", "obj_path": out_path}


def main():
    parser = argparse.ArgumentParser(
        description="Generate canopy field OBJs from a Sobol catalog (chunkable for multi-node runs)."
    )
    parser.add_argument("--catalog", type=str, default=CATALOG_PATH,
                        help="Path to sim catalog (.csv or .parquet)")
    parser.add_argument("--base_json", type=str, required=True,
                        help="Base NURBS JSON template path")
    parser.add_argument("--out_dir", type=str, default="canopies",
                        help="Output directory for OBJ files")

    parser.add_argument("--start_id", type=int, default=0,
                        help="First simulation_id (inclusive)")
    parser.add_argument("--end_id", type=int, default=None,
                        help="End simulation_id (exclusive). If omitted, uses start_id + n_sims.")
    parser.add_argument("--n_sims", type=int, default=None,
                        help="Number of sims to run starting at start_id. Used if end_id is not given.")

    parser.add_argument("--row_spacing", type=float, default=0.76)
    parser.add_argument("--plant_spacing", type=float, default=0.20)

    parser.add_argument("--rotation_csv", type=str, default=None,
                        help="Optional CSV to append per-plant rotation angles")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip sim_ids whose field_cropped OBJ already exists and is non-empty")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_catalog(args.catalog)

    if args.end_id is None:
        if args.n_sims is None:
            raise ValueError("Provide either --end_id or --n_sims.")
        start_id = args.start_id
        end_id = args.start_id + args.n_sims
    else:
        start_id = args.start_id
        end_id = args.end_id
        if args.n_sims is not None and (start_id + args.n_sims) != end_id:
            raise ValueError("If both --end_id and --n_sims are provided, they must be consistent.")

    max_id = int(df["simulation_id"].max())
    if end_id - 1 > max_id:
        raise ValueError(f"Requested end_id={end_id} exceeds max simulation_id in catalog ({max_id}).")

    for sim_id in range(start_id, end_id):
        status = generate_one_canopy(
            sim_id=sim_id,
            base_json_path=args.base_json,
            out_dir=args.out_dir,
            df=df,
            row_spacing=args.row_spacing,
            plant_spacing=args.plant_spacing,
            rotation_csv_path=args.rotation_csv,
            skip_existing=args.skip_existing,
        )
        print(status)


if __name__ == "__main__":
    main()
