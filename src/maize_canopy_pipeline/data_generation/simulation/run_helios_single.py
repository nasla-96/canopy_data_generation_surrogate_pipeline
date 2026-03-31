#!/usr/bin/env python3
import os
import shutil
import math
import json
import argparse
import configparser
import subprocess
from typing import Dict, Optional, List, Tuple

from ..catalog.generate_catalog import main as generate_catalog_main
from ..geometry.generate_canopies import load_catalog, generate_one_canopy

import numpy as np
import pandas as pd
from typing import Optional

CATALOG_PATH_DEFAULT = "catalog/sim_catalog.csv"


def extract_PAR_from_output(stdout: str) -> Optional[float]:
    """Parse a line like: 'PAR_VALUE: 123.456' from Helios stdout."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("PAR_VALUE:"):
            try:
                return float(line.split("PAR_VALUE:")[-1].strip())
            except ValueError:
                return None
    return None


def run_helios(
    config_file_path: str,
    device_id: int,
    helios_build_dir: str,
    modules: Optional[List[str]] = None,
    use_modules: bool = True,
) -> Optional[float]:
    """Run a single Helios PAR simulation on the specified GPU."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        module_cmd = ""
        if use_modules and modules:
            module_cmd = " && ".join([f"module load {m}" for m in modules]) + " && "

        cfg_abs = os.path.abspath(config_file_path)
        inner = f"{module_cmd}./par_parallel {cfg_abs}"
        cmd = "bash -lc " + repr(inner)

        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=helios_build_dir,
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print("Helios execution failed with return code:", process.returncode)
            print("stderr:", stderr.decode(errors="replace"))
            return None

        return extract_PAR_from_output(stdout.decode(errors="replace"))

    except Exception as e:
        print("An error occurred while running Helios:", str(e))
        return None


def update_config_file(
    config_file_path: str,
    field_path: str,
    lat_deg: float,
    lon_deg: float,
    utc_offset: int,
) -> None:
    """Update the Helios INI with the OBJ path and location."""
    config = configparser.ConfigParser()
    config.read(config_file_path)

    if not config.has_section("Paths"):
        config.add_section("Paths")
    config.set("Paths", "fieldfilepath", os.path.abspath(field_path))

    if not config.has_section("Location"):
        config.add_section("Location")
    config.set("Location", "latitude", str(lat_deg))
    config.set("Location", "longitude", str(lon_deg))
    config.set("Location", "utc_offset", str(utc_offset))

    with open(config_file_path, "w") as configfile:
        config.write(configfile)


def _read_catalog(catalog_path: str) -> pd.DataFrame:
    ext = os.path.splitext(catalog_path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(catalog_path)
    return pd.read_csv(catalog_path)


def _read_latlon_from_catalog(sim_id: int, catalog_path: str) -> Optional[Tuple[float, float]]:
    """Fetch (lat_deg, lon_deg) for a simulation_id from the catalog."""
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    df = _read_catalog(catalog_path)
    row = df.loc[df["simulation_id"] == sim_id]
    if row.empty:
        return None

    row = row.iloc[0]
    return float(row["lat_deg"]), float(row["lon_deg"])


def _ensure_catalog_exists(
    n_sims: int,
    catalog_path: str,
    lon_deg: float,
    append: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, str]:
    """Create catalog if missing (uses generate_catalog.py's main())."""
    if os.path.exists(catalog_path) and os.path.getsize(catalog_path) > 0 and not append:
        return {"status": "SKIPPED", "catalog_path": catalog_path}

    os.makedirs(os.path.dirname(catalog_path) or ".", exist_ok=True)

    from generate_catalog import main as generate_catalog_main

    generate_catalog_main(n_sims=n_sims, out_path=catalog_path, lon_deg_fixed=lon_deg, append=append, seed=seed)
    if not (os.path.exists(catalog_path) and os.path.getsize(catalog_path) > 0):
        return {"status": "FAILED", "catalog_path": catalog_path, "error": "catalog_write_failed"}

    return {"status": "SUCCESS", "catalog_path": catalog_path}


def _ensure_canopy_obj(
    sim_id: int,
    canopy_dir: str,
    base_json_path: Optional[str],
    catalog_path: str,
    row_spacing: float,
    plant_spacing: float,
    rotation_csv_path: Optional[str] = None,
    skip_existing: bool = False,
) -> Dict[str, str]:
    """Ensure an OBJ exists for sim_id; if missing, generate it."""
    candidates = [
        os.path.join(canopy_dir, f"sim_{sim_id:06d}_field_cropped.obj"),
        os.path.join(canopy_dir, f"sim_{sim_id:06d}.obj"),
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return {"status": "SUCCESS", "obj_path": p}

    if base_json_path is None:
        return {
            "status": "FAILED",
            "error": "obj_not_found_and_no_base_json_path_provided",
            "obj_path": candidates[0],
        }


    if not os.path.exists(catalog_path):
        return {
            "status": "FAILED",
            "error": f"catalog_not_found: {catalog_path}",
            "obj_path": candidates[0],
        }

    df = load_catalog(catalog_path)
    os.makedirs(canopy_dir, exist_ok=True)

    status = generate_one_canopy(
        sim_id=sim_id,
        base_json_path=base_json_path,
        out_dir=canopy_dir,
        df=df,
        row_spacing=row_spacing,
        plant_spacing=plant_spacing,
        rotation_csv_path=rotation_csv_path,
        skip_existing=skip_existing,
    )

    if status.get("status") not in ("SUCCESS", "SKIPPED"):
        return {
            "status": "FAILED",
            "error": status.get("error", "unknown_canopy_generation_error"),
            "obj_path": candidates[0],
        }

    obj_path = status.get("obj_path", "")
    if not obj_path or not (os.path.exists(obj_path) and os.path.getsize(obj_path) > 0):
        return {
            "status": "FAILED",
            "error": "generated_obj_missing_or_empty",
            "obj_path": obj_path or candidates[0],
        }

    return {"status": "SUCCESS", "obj_path": obj_path}


def run_helios_single(
    sim_id: int,
    device_id: int,
    base_config_template_path: str,
    canopy_dir: str = "canopies",
    config_out_dir: str = "configs",
    catalog_path: str = CATALOG_PATH_DEFAULT,
    base_json_path: Optional[str] = None,
    utc_offset: int = 5,
    row_spacing: float = 0.76,
    plant_spacing: float = 0.20,
    rotation_csv_path: Optional[str] = None,
    skip_existing_obj: bool = False,
    helios_build_dir: str = "/home/ubuntu/Helios/samples/par_parallel/build",
    modules: Optional[List[str]] = None,
    use_modules: bool = True,
) -> Dict:
    """End-to-end single-simulation runner."""

    obj_status = _ensure_canopy_obj(
        sim_id,
        canopy_dir=canopy_dir,
        base_json_path=base_json_path,
        catalog_path=catalog_path,
        row_spacing=row_spacing,
        plant_spacing=plant_spacing,
        rotation_csv_path=rotation_csv_path,
        skip_existing=skip_existing_obj,
    )
    if obj_status.get("status") != "SUCCESS":
        return {
            "simulation_id": sim_id,
            "status": "FAILED",
            "error": obj_status.get("error", "obj_not_found"),
            "net_PAR": math.nan,
            "obj_path": obj_status.get("obj_path", ""),
            "config_path": "",
        }

    obj_path = obj_status["obj_path"]

    latlon = _read_latlon_from_catalog(sim_id, catalog_path=catalog_path)
    if latlon is None:
        return {
            "simulation_id": sim_id,
            "status": "FAILED",
            "error": "sim_id_not_in_catalog",
            "net_PAR": math.nan,
            "obj_path": obj_path,
            "config_path": "",
        }
    lat_deg, lon_deg = latlon

    if not os.path.exists(base_config_template_path):
        return {
            "simulation_id": sim_id,
            "status": "FAILED",
            "error": f"base_config_template_not_found: {base_config_template_path}",
            "net_PAR": math.nan,
            "obj_path": obj_path,
            "config_path": "",
        }

    os.makedirs(config_out_dir, exist_ok=True)
    config_path = os.path.join(config_out_dir, f"sim_{sim_id:06d}.ini")
    shutil.copy(base_config_template_path, config_path)
    update_config_file(config_path, obj_path, lat_deg, lon_deg, utc_offset)

    net_par = run_helios(
        config_file_path=config_path,
        device_id=device_id,
        helios_build_dir=helios_build_dir,
        modules=modules,
        use_modules=use_modules,
    )

    if net_par is None or not np.isfinite(net_par) or net_par < 0:
        return {
            "simulation_id": sim_id,
            "status": "FAILED",
            "error": "invalid_or_missing_PAR",
            "net_PAR": math.nan,
            "obj_path": obj_path,
            "config_path": config_path,
        }

    return {
        "simulation_id": sim_id,
        "status": "SUCCESS",
        "error": "",
        "net_PAR": float(net_par),
        "obj_path": obj_path,
        "config_path": config_path,
    }


def main_cli():
    p = argparse.ArgumentParser(description="End-to-end single sim: (optional) catalog -> (optional) OBJ -> Helios")
    p.add_argument("--sim_id", type=int, default=0)
    p.add_argument("--device_id", type=int, default=0)

    p.add_argument("--catalog", type=str, default=CATALOG_PATH_DEFAULT)
    p.add_argument("--ensure_catalog", action="store_true", help="If set, create catalog if missing.")
    p.add_argument("--append_catalog", action="store_true", help="If set with --ensure_catalog, append NEW rows.")
    p.add_argument("--n_sims", type=int, default=1, help="Used only with --ensure_catalog (NEW rows).")
    p.add_argument("--lon_deg", type=float, default=93.62)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--base_json", type=str, default=None,
                   help="If OBJ missing, this is required to generate it.")
    p.add_argument("--canopy_dir", type=str, default="canopies")
    p.add_argument("--row_spacing", type=float, default=0.76)
    p.add_argument("--plant_spacing", type=float, default=0.20)
    p.add_argument("--rotation_csv", type=str, default=None)
    p.add_argument("--skip_existing_obj", action="store_true")

    p.add_argument("--ini_template", type=str, required=True)
    p.add_argument("--config_out_dir", type=str, default="configs")
    p.add_argument("--utc_offset", type=int, default=5)

    p.add_argument("--helios_build_dir", type=str,
                   default="/work/mech-ai-scratch/nasla/helios_new/Helios/samples/par_parallel/build")
    p.add_argument("--no_modules", action="store_true", help="Disable module loading.")
    p.add_argument("--modules", type=str, nargs="*",
                   default=["mesa-glu", "boost/1.86.0-gvbpdbb"],
                   help="Modules to load before running par_parallel.")

    args = p.parse_args()

    if args.ensure_catalog:
        cat_status = _ensure_catalog_exists(
            n_sims=args.n_sims,
            catalog_path=args.catalog,
            lon_deg=args.lon_deg,
            append=args.append_catalog,
            seed=args.seed,
        )
        if cat_status.get("status") == "FAILED":
            print(json.dumps(cat_status, indent=2))
            raise SystemExit(1)

    result = run_helios_single(
        sim_id=args.sim_id,
        device_id=args.device_id,
        base_config_template_path=args.ini_template,
        canopy_dir=args.canopy_dir,
        config_out_dir=args.config_out_dir,
        catalog_path=args.catalog,
        base_json_path=args.base_json,
        utc_offset=args.utc_offset,
        row_spacing=args.row_spacing,
        plant_spacing=args.plant_spacing,
        rotation_csv_path=args.rotation_csv,
        skip_existing_obj=args.skip_existing_obj,
        helios_build_dir=args.helios_build_dir,
        modules=args.modules,
        use_modules=(not args.no_modules),
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main_cli()
