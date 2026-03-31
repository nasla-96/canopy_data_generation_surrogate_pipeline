#!/usr/bin/env python3
"""
scripts/run_pipeline.py

End-to-end wrapper for the Helios surrogate pipeline (UPDATED CSV WORKFLOW):

1) (optional) Extend Sobol catalog (CSV)
2) Generate canopy OBJs for a sim range
3) Run Helios for the same sim range and save results (CSV)
4) (optional) Write scaled catalog CSV + merged ML dataset CSV (params + net_PAR)

Restart-safe:
- --skip_existing_obj skips OBJ generation if canopy already exists
- --skip_existing_results skips Helios runs if results already exist for a sim_id with SUCCESS

Works on OCI (no `module` command) by passing --no_modules.
"""
import os, sys

from ..catalog.generate_catalog import main as generate_catalog_main
from ..geometry.generate_canopies import load_catalog, generate_one_canopy
from ..simulation.run_helios_single import run_helios_single as run_one_helios
from ..dataset.make_scaled_catalog import main as make_scaled_main
from ..dataset.build_ml_dataset import main as build_ml_main

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import os
import json
import argparse
from typing import List, Dict, Tuple, Set, Optional


import numpy as np
import pandas as pd
import csv


def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return [0]
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _read_results_success_ids(results_csv: str) -> Set[int]:
    if not os.path.exists(results_csv) or os.path.getsize(results_csv) == 0:
        return set()
    try:
        prev = pd.read_csv(results_csv)
        if "simulation_id" in prev.columns and "status" in prev.columns:
            ok = prev.loc[prev["status"] == "SUCCESS", "simulation_id"].astype(int).tolist()
            return set(ok)
    except Exception:
        pass
    return set()

def _append_result_row(results_csv: str, row: dict, fieldnames: Optional[List[str]]) -> List[str]:
    """
    Append a single result row to results_csv (creates file + header if missing).
    Keeps a stable column order using fieldnames from the first row written.
    """
    _ensure_dir(results_csv)

    if fieldnames is None:
        fieldnames = list(row.keys())

    file_exists = os.path.exists(results_csv) and os.path.getsize(results_csv) > 0

    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()

        safe_row = {k: row.get(k, "") for k in fieldnames}
        writer.writerow(safe_row)

        # crash-safe flush
        f.flush()
        os.fsync(f.fileno())

    return fieldnames
    
def main():
    p = argparse.ArgumentParser(
        description="End-to-end pipeline: catalog -> canopy OBJs -> Helios runs (results CSV) + optional ML merge"
    )

    # Overall sim range
    p.add_argument("--start_id", type=int, default=0, help="First simulation_id (inclusive).")
    p.add_argument("--n_sims", type=int, default=10, help="Number of sims to run from start_id.")
    p.add_argument("--end_id", type=int, default=None,
                   help="End simulation_id (exclusive). If set, overrides --n_sims.")

    # Catalog (CSV)
    p.add_argument("--catalog", type=str, default="catalog/sim_catalog.csv")
    p.add_argument("--extend_catalog", action="store_true",
                   help="Append NEW rows to the catalog before running. Uses --catalog_n_new and --lon_deg.")
    p.add_argument("--catalog_n_new", type=int, default=0,
                   help="How many NEW sims to append when --extend_catalog is set.")
    p.add_argument("--seed", type=int, default=None,
                   help="Optional seed for the NEW scrambled Sobol batch (only for appended rows).")
    p.add_argument("--lon_deg", type=float, default=93.62,
                   help="Fixed longitude used for catalog generation (Ames default).")

    # Geometry / canopy generation
    p.add_argument("--base_json", type=str, required=True, help="Base NURBS JSON template path.")
    p.add_argument("--canopy_dir", type=str, default="canopies/test_10")
    p.add_argument("--row_spacing", type=float, default=0.76)
    p.add_argument("--plant_spacing", type=float, default=0.20)
    p.add_argument("--rotation_csv", type=str, default=None)
    p.add_argument("--skip_existing_obj", action="store_true",
                   help="Skip OBJ generation if canopy OBJ already exists.")

    # Helios run
    p.add_argument("--ini_template", type=str, required=True,
                   help="Base Helios INI template to copy per sim.")
    p.add_argument("--config_out_dir", type=str, default="configs/test_10")
    p.add_argument("--utc_offset", type=int, default=5)
    p.add_argument("--helios_build_dir", type=str, required=True,
                   help="Directory where the Helios executable is run from (contains ./par_parallel).")

    # GPUs / modules
    p.add_argument("--device_ids", type=str, default="0",
                   help='Comma-separated GPU ids to round-robin, e.g. "0" or "0,1,2,3".')
    p.add_argument("--no_modules", action="store_true",
                   help="Disable module loading (recommended on OCI where `module` is unavailable).")
    p.add_argument("--modules", type=str, nargs="*", default=None,
                   help="Optional modules to load (HPC only).")

    # Results (CSV only)
    p.add_argument("--results_csv", type=str, default="results/test_10_results.csv")
    p.add_argument("--skip_existing_results", action="store_true",
                   help="If results CSV exists, skip sims already present with SUCCESS status.")

    # ML artifacts (optional)
    p.add_argument("--write_ml_files", action="store_true",
                   help="If set, writes scaled catalog CSV and merged ML dataset CSV after runs.")
    p.add_argument("--scaled_catalog_csv", type=str, default="catalog/sim_catalog_scaled.csv")
    p.add_argument("--ml_dataset_csv", type=str, default="results/ml_dataset.csv")

    args = p.parse_args()

    # Resolve sim range
    if args.end_id is None:
        start_id = args.start_id
        end_id = args.start_id + args.n_sims
    else:
        start_id = args.start_id
        end_id = args.end_id

    if end_id <= start_id:
        raise SystemExit(f"Invalid range: start_id={start_id}, end_id={end_id}")

    device_ids = _parse_int_list(args.device_ids) or [0]

    # (0) Optionally extend catalog
    if args.extend_catalog:
        if args.catalog_n_new <= 0:
            raise SystemExit("--extend_catalog requires --catalog_n_new > 0")
        _ensure_dir(args.catalog)
        generate_catalog_main(
            n_sims=int(args.catalog_n_new),
            out_path=args.catalog,
            lon_deg_fixed=float(args.lon_deg),
            append=True,
            seed=args.seed,
        )

    if not os.path.exists(args.catalog):
        raise SystemExit(f"Catalog not found: {args.catalog}")

    # Load catalog once
    df = load_catalog(args.catalog)
    max_id = int(df["simulation_id"].max())
    if end_id - 1 > max_id:
        raise SystemExit(
            f"Requested end_id={end_id} exceeds max simulation_id in catalog ({max_id}). "
            f"Extend the catalog or lower end_id."
        )

    # Existing results (restart-safe)
    existing_success = _read_results_success_ids(args.results_csv) if args.skip_existing_results else set()

    os.makedirs(args.canopy_dir, exist_ok=True)
    os.makedirs(args.config_out_dir, exist_ok=True)
    _ensure_dir(args.results_csv)

    # results: List[Dict] = []

    # for k, sim_id in enumerate(range(start_id, end_id)):
    #     if sim_id in existing_success:
    #         continue

    #     device_id = device_ids[k % len(device_ids)]

    #     geom_status = generate_one_canopy(
    #         sim_id=sim_id,
    #         base_json_path=args.base_json,
    #         out_dir=args.canopy_dir,
    #         df=df,
    #         row_spacing=args.row_spacing,
    #         plant_spacing=args.plant_spacing,
    #         rotation_csv_path=args.rotation_csv,
    #         skip_existing=args.skip_existing_obj,
    #     )

    #     if geom_status.get("status") not in ("SUCCESS", "SKIPPED"):
    #         out = {
    #             "simulation_id": sim_id,
    #             "status": "FAILED",
    #             "stage": "GEOMETRY",
    #             "error": geom_status.get("error", "unknown_geometry_error"),
    #             "net_PAR": np.nan,
    #             "obj_path": geom_status.get("obj_path", ""),
    #             "config_path": "",
    #             "device_id": device_id,
    #         }
    #         print(json.dumps(out, indent=2))
    #         results.append(out)
    #         continue

    #     run_out = run_one_helios(
    #         sim_id=sim_id,
    #         device_id=device_id,
    #         base_config_template_path=args.ini_template,
    #         canopy_dir=args.canopy_dir,
    #         config_out_dir=args.config_out_dir,
    #         catalog_path=args.catalog,
    #         base_json_path=args.base_json,
    #         utc_offset=args.utc_offset,
    #         row_spacing=args.row_spacing,
    #         plant_spacing=args.plant_spacing,
    #         rotation_csv_path=args.rotation_csv,
    #         skip_existing_obj=True,
    #         helios_build_dir=args.helios_build_dir,
    #         modules=args.modules,
    #         use_modules=(not args.no_modules),
    #     )

    #     run_out["device_id"] = device_id
    #     run_out["stage"] = "HELIOS"
    #     print(json.dumps(run_out, indent=2))
    #     results.append(run_out)

    # # Save / append results (CSV)
    # if results:
    #     out_df = pd.DataFrame(results)

    #     if os.path.exists(args.results_csv) and os.path.getsize(args.results_csv) > 0:
    #         try:
    #             prev = pd.read_csv(args.results_csv)
    #             out_df = pd.concat([prev, out_df], ignore_index=True)
    #             out_df = out_df.sort_values(["simulation_id"]).drop_duplicates(
    #                 subset=["simulation_id"], keep="last"
    #             )
    #         except Exception:
    #             pass

    #     out_df.to_csv(args.results_csv, index=False)
    #     print(f"Saved results CSV: {args.results_csv}")
    # else:
    #     print("No new simulations were run (everything was already complete or skipped).")

    csv_fieldnames: Optional[List[str]] = None

    for k, sim_id in enumerate(range(start_id, end_id)):
        if sim_id in existing_success:
            continue

        device_id = device_ids[k % len(device_ids)]

        geom_status = generate_one_canopy(
            sim_id=sim_id,
            base_json_path=args.base_json,
            out_dir=args.canopy_dir,
            df=df,
            row_spacing=args.row_spacing,
            plant_spacing=args.plant_spacing,
            rotation_csv_path=args.rotation_csv,
            skip_existing=args.skip_existing_obj,
        )

        if geom_status.get("status") not in ("SUCCESS", "SKIPPED"):
            out = {
                "simulation_id": sim_id,
                "status": "FAILED",
                "stage": "GEOMETRY",
                "error": geom_status.get("error", "unknown_geometry_error"),
                "net_PAR": np.nan,
                "obj_path": geom_status.get("obj_path", ""),
                "config_path": "",
                "device_id": device_id,
            }
            print(json.dumps(out, indent=2))
            csv_fieldnames = _append_result_row(args.results_csv, out, csv_fieldnames)
            continue

        run_out = run_one_helios(
            sim_id=sim_id,
            device_id=device_id,
            base_config_template_path=args.ini_template,
            canopy_dir=args.canopy_dir,
            config_out_dir=args.config_out_dir,
            catalog_path=args.catalog,
            base_json_path=args.base_json,
            utc_offset=args.utc_offset,
            row_spacing=args.row_spacing,
            plant_spacing=args.plant_spacing,
            rotation_csv_path=args.rotation_csv,
            skip_existing_obj=True,
            helios_build_dir=args.helios_build_dir,
            modules=args.modules,
            use_modules=(not args.no_modules),
        )

        run_out["device_id"] = device_id
        run_out["stage"] = "HELIOS"
        print(json.dumps(run_out, indent=2))

        csv_fieldnames = _append_result_row(args.results_csv, run_out, csv_fieldnames)

    print(f"Streaming results written to: {args.results_csv}")

# Optional: write scaled catalog + ML dataset
    if args.write_ml_files:
        from make_scaled_catalog import main as make_scaled_main
        from build_ml_dataset import main as build_ml_main

        make_scaled_main(args.catalog, args.scaled_catalog_csv, args.base_json)
        build_ml_main(args.scaled_catalog_csv, args.results_csv, args.ml_dataset_csv)


if __name__ == "__main__":
    main()
