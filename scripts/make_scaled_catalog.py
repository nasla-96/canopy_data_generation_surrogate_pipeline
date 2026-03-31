#!/usr/bin/env python3
"""
scripts/make_scaled_catalog.py

Create a *scaled* catalog CSV from the raw catalog (CSV or Parquet), using the
same mappings/logic as canopy_from_sobol.py, including the biologically
constrained inter-leaf placement logic that depends on the base JSON stalk height.

Outputs a CSV with:
- simulation_id, lat_deg, lon_deg
- stalk_scale
- interleaf_pos_{i}  (normalized 0..1 positions actually used)
- length_cm_{i}, width_cm_{i}
- theta_deg_{i}, phi_deg_{i}
- curv_{i}, twist_{i}

This is designed specifically for your ML input table.
"""

import os
import argparse
import numpy as np
import pandas as pd
from geomdl import exchange, multi

NUM_LEAVES = 12

def map_to_range(x, lo, hi):
    arr = np.asarray(x, dtype=float)
    return lo + arr * (hi - lo)

def read_catalog(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def compute_base_stalk_height_cm(base_json_path: str) -> float:
    data = exchange.import_json(base_json_path)
    surf_cont = multi.SurfaceContainer(data)
    stalk_pts = surf_cont[0].ctrlpts
    min_z = min(p[2] for p in stalk_pts)
    max_z = max(p[2] for p in stalk_pts)
    return float(max_z - min_z)

def interleaf_positions_from_raw(raw_interleaf: np.ndarray, stalk_scale: float, base_stalk_h_cm: float) -> np.ndarray:
    """Replicates canopy_from_sobol.py's inter-leaf spacing -> positions logic."""
    min_spacing_cm = 7.0
    max_spacing_cm = 20.0

    min_leaf_offset_cm = 13.0
    top_margin_cm = 15.0

    scaled_stalk_h = base_stalk_h_cm * stalk_scale
    available_z_span = (scaled_stalk_h - top_margin_cm) - min_leaf_offset_cm

    if available_z_span <= 1e-6:
        # fall back to unconstrained positions (still sort)
        return np.sort(np.clip(raw_interleaf, 0.0, 1.0))

    n = raw_interleaf.size
    target_sum = available_z_span

    min_feasible = target_sum / (n - 1)
    if (n - 1) * min_spacing_cm > target_sum:
        min_spacing = min_feasible
        max_spacing = min_feasible
    elif (n - 1) * max_spacing_cm < target_sum:
        min_spacing = min_feasible
        max_spacing = min_feasible
    else:
        min_spacing = min_spacing_cm
        max_spacing = max_spacing_cm

    u = np.asarray(raw_interleaf[: n - 1], dtype=float)
    spacings = min_spacing + u * (max_spacing - min_spacing)
    spacings = np.clip(spacings, min_spacing, max_spacing)

    for _ in range(20):
        delta = target_sum - float(np.sum(spacings))
        if abs(delta) < 1e-6:
            break
        room = (max_spacing - spacings) if delta > 0 else (spacings - min_spacing)
        room_sum = float(np.sum(room))
        if room_sum <= 1e-12:
            break
        spacings = spacings + delta * (room / room_sum)
        spacings = np.clip(spacings, min_spacing, max_spacing)

    attachment_rel = np.insert(np.cumsum(spacings), 0, 0.0)  # length n
    pos = attachment_rel / target_sum
    pos = np.clip(pos, 0.0, 1.0)
    return pos

def main(in_catalog: str, out_catalog: str, base_json: str) -> None:
    df = read_catalog(in_catalog)
    base_stalk_h_cm = compute_base_stalk_height_cm(base_json)

    records = []
    for _, row in df.iterrows():
        sim_id = int(row["simulation_id"])

        # Raw vectors
        raw_stalk = float(row["stalk_raw"])
        raw_interleaf = np.array([row[f"interleaf_raw_{i}"] for i in range(NUM_LEAVES)], dtype=float)
        raw_length    = np.array([row[f"length_raw_{i}"]    for i in range(NUM_LEAVES)], dtype=float)
        raw_width     = np.array([row[f"width_raw_{i}"]     for i in range(NUM_LEAVES)], dtype=float)
        raw_theta     = np.array([row[f"theta_raw_{i}"]     for i in range(NUM_LEAVES)], dtype=float)
        raw_phi       = np.array([row[f"phi_raw_{i}"]       for i in range(NUM_LEAVES)], dtype=float)
        raw_curv      = np.array([row[f"curv_raw_{i}"]      for i in range(NUM_LEAVES)], dtype=float)
        raw_twist     = np.array([row[f"twist_raw_{i}"]     for i in range(NUM_LEAVES)], dtype=float)

        # Scaled mappings (must match canopy_from_sobol.py)
        stalk_scale = float(map_to_range(raw_stalk, 0.7, 1.3))
        interleaf_pos = interleaf_positions_from_raw(raw_interleaf, stalk_scale, base_stalk_h_cm)

        lengths = map_to_range(raw_length, 40.0, 110.0)   # cm
        widths  = map_to_range(raw_width, 5.0, 14.0)      # cm
        thetas  = map_to_range(raw_theta, -40.0, 40.0)    # deg
        phis    = map_to_range(raw_phi, 0.0, 360.0)       # deg
        curv    = map_to_range(raw_curv, 0.0, 2.0)
        twist   = map_to_range(raw_twist, -5.0, 5.0)

        out = {
            "simulation_id": sim_id,
            "lat_deg": float(row["lat_deg"]),
            "lon_deg": float(row["lon_deg"]),
            "stalk_scale": stalk_scale,
        }
        for i in range(NUM_LEAVES):
            out[f"interleaf_pos_{i}"] = float(interleaf_pos[i])
            out[f"length_cm_{i}"] = float(lengths[i])
            out[f"width_cm_{i}"] = float(widths[i])
            out[f"theta_deg_{i}"] = float(thetas[i])
            out[f"phi_deg_{i}"] = float(phis[i])
            out[f"curv_{i}"] = float(curv[i])
            out[f"twist_{i}"] = float(twist[i])

        records.append(out)

    out_df = pd.DataFrame.from_records(records)
    os.makedirs(os.path.dirname(out_catalog) or ".", exist_ok=True)
    out_df.to_csv(out_catalog, index=False)
    print(f"Wrote scaled catalog CSV: {out_catalog}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_catalog", required=True)
    ap.add_argument("--out_catalog", required=True)
    ap.add_argument("--base_json", required=True, help="Base NURBS JSON used to compute stalk height banding.")
    args = ap.parse_args()
    main(args.in_catalog, args.out_catalog, args.base_json)
