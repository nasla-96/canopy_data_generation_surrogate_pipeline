import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Merge scaled parameters with Helios PAR results for ML dataset.")
    ap.add_argument("--params_csv", default="catalog/sim_catalog_scaled.csv")
    ap.add_argument("--results_csv", default="results/run_2500.csv")
    ap.add_argument("--out_csv", default="results/ml_dataset_2500.csv")
    ap.add_argument("--only_success", action="store_true",
                    help="Keep only rows with status==SUCCESS and finite net_PAR.")
    ap.add_argument("--drop_failures", action="store_true",
                    help="Drop rows with missing net_PAR (even if status missing).")
    args = ap.parse_args()

    # Read inputs
    params = pd.read_csv(args.params_csv)
    results = pd.read_csv(args.results_csv)

    # Basic schema checks
    if "simulation_id" not in params.columns:
        raise ValueError(f"'simulation_id' missing from params_csv: {args.params_csv}")
    if "simulation_id" not in results.columns:
        raise ValueError(f"'simulation_id' missing from results_csv: {args.results_csv}")

    # Ensure numeric simulation_id
    params["simulation_id"] = params["simulation_id"].astype(int)
    results["simulation_id"] = results["simulation_id"].astype(int)

    # If results has duplicates (reruns), keep the last occurrence per simulation_id
    # (Assumes file order is append order; if you have a timestamp column, we can sort by it instead.)
    results = results.drop_duplicates(subset=["simulation_id"], keep="last")

    # Keep only the useful output columns if present
    keep_cols = ["simulation_id"]
    for c in ["net_PAR", "status", "error", "obj_path", "config_path", "device_id"]:
        if c in results.columns:
            keep_cols.append(c)
    results = results[keep_cols]

    # Merge (left join keeps all parameter rows; you can switch to inner if you prefer)
    merged = params.merge(results, on="simulation_id", how="left")

    # Normalize/clean net_PAR if present
    if "net_PAR" in merged.columns:
        merged["net_PAR"] = pd.to_numeric(merged["net_PAR"], errors="coerce")

    # Optional filtering for ML-ready dataset
    if args.only_success:
        if "status" not in merged.columns:
            raise ValueError("--only_success requested but 'status' column not found in results_csv.")
        merged = merged.loc[(merged["status"] == "SUCCESS") & np.isfinite(merged["net_PAR"])]

    if args.drop_failures:
        if "net_PAR" not in merged.columns:
            raise ValueError("--drop_failures requested but 'net_PAR' column not found in results_csv.")
        merged = merged.loc[np.isfinite(merged["net_PAR"])]

    # Save
    merged.to_csv(args.out_csv, index=False)

    # Quick summary
    n_total = len(merged)
    n_par = merged["net_PAR"].notna().sum() if "net_PAR" in merged.columns else 0
    print(f"Wrote: {args.out_csv}")
    print(f"Rows: {n_total}")
    if "net_PAR" in merged.columns:
        print(f"Rows with net_PAR present: {n_par}")


if __name__ == "__main__":
    main()