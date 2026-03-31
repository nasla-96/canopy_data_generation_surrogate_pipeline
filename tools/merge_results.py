#!/usr/bin/env python3
"""
merge_results.py

Merge multiple Helios results CSVs with potentially different column orders,
normalize to a standard schema, de-duplicate by simulation_id, sort, and report gaps.

Example:
  python merge_results.py \
    --inputs ../results/run_0_15_36k_50k_results.csv ../results/merged_results_5000_33000.csv \
    --out ../results/results_merged_dedup_sorted.csv

Optional:
  --expected-gap 33000 35999
  --show-gaps 200
  --dedup-policy best   (default)  # best status/net_PAR/error heuristic
  --dedup-policy prefer-last       # later inputs override earlier for duplicate IDs
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd


COL_ORDER = [
    "simulation_id",
    "net_PAR",
    "status",
    "error",
    "obj_path",
    "config_path",
    "device_id",
    "stage",
]


def read_and_normalize(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    for c in COL_ORDER:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[COL_ORDER].copy()
    df["simulation_id"] = pd.to_numeric(df["simulation_id"], errors="coerce").astype("Int64")
    df["net_PAR"] = pd.to_numeric(df["net_PAR"], errors="coerce")
    return df


def status_rank(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()

    success_tokens = {"ok", "success", "succeeded", "done", "complete", "completed", "finished", "0"}
    neutral_tokens = {"running", "in_progress", "in progress", "started", "pending"}
    fail_tokens = {"fail", "failed", "error", "crash", "killed", "timeout"}

    rank = pd.Series(np.zeros(len(s), dtype=int), index=s.index)
    rank[s.isin(success_tokens)] = 3
    rank[s.isin(neutral_tokens)] = 2
    rank[s.isin(fail_tokens)] = 1
    rank[s.isin({"nan", "none", "<na>", ""})] = 0
    return rank


def dedup_best(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep "best" row per simulation_id:
      1) higher status_rank
      2) has net_PAR
      3) shorter/empty error preferred
    """
    df = df.copy()
    df["_rank"] = status_rank(df["status"])
    df["_has_par"] = df["net_PAR"].notna().astype(int)
    err = df["error"].astype(str).replace("nan", "")
    df["_err_len"] = err.str.len()

    df = df.sort_values(
        by=["simulation_id", "_rank", "_has_par", "_err_len"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    out = df.drop_duplicates(subset=["simulation_id"], keep="first").copy()
    out = out.drop(columns=["_rank", "_has_par", "_err_len"], errors="ignore")
    out = out.sort_values("simulation_id", kind="mergesort").reset_index(drop=True)
    return out


def dedup_prefer_last(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer later input files for duplicates by simulation_id.
    Requires a column _src_order where higher = later.
    """
    df = df.sort_values(by=["simulation_id", "_src_order"], ascending=[True, True], kind="mergesort")
    out = df.drop_duplicates(subset=["simulation_id"], keep="last").copy()
    out = out.drop(columns=["_src_order"], errors="ignore")
    out = out.sort_values("simulation_id", kind="mergesort").reset_index(drop=True)
    return out


def find_gap_ranges(ids):
    ids = np.asarray(ids, dtype=int)
    if ids.size == 0:
        return []
    missing = []
    diffs = np.diff(ids)
    gap_starts_idx = np.where(diffs > 1)[0]
    for i in gap_starts_idx:
        start = int(ids[i] + 1)
        end = int(ids[i + 1] - 1)
        missing.append((start, end))
    return missing


def main():
    ap = argparse.ArgumentParser(description="Merge/dedup/sort Helios results CSVs and report simulation_id gaps.")
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input CSV paths (space-separated). Later files can override earlier with --dedup-policy prefer-last.",
    )
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument(
        "--dedup-policy",
        choices=["best", "prefer-last"],
        default="best",
        help="How to resolve duplicates: 'best' (status/net_PAR/error heuristic) or 'prefer-last' (later inputs win).",
    )
    ap.add_argument(
        "--expected-gap",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Optional expected gap range to check, inclusive (e.g., 33000 35999).",
    )
    ap.add_argument(
        "--show-gaps",
        type=int,
        default=50,
        help="How many gap ranges to print (default: 50).",
    )
    args = ap.parse_args()

    # Validate paths
    inputs = [Path(p) for p in args.inputs]
    for p in inputs:
        if not p.exists():
            print(f"ERROR: Could not find input file: {p}", file=sys.stderr)
            sys.exit(1)

    # Read + normalize
    dfs = []
    for i, p in enumerate(inputs):
        d = read_and_normalize(str(p))
        if args.dedup_policy == "prefer-last":
            d["_src_order"] = i
        dfs.append(d)

    combined = pd.concat(dfs, ignore_index=True)

    # Drop rows with missing simulation_id
    combined = combined[combined["simulation_id"].notna()].copy()
    combined["simulation_id"] = combined["simulation_id"].astype(int)

    # Duplicate stats (pre-dedup)
    dup_counts = combined["simulation_id"].value_counts()
    num_dup_ids = int((dup_counts > 1).sum())
    extra_dup_rows = int(dup_counts[dup_counts > 1].sum() - (dup_counts > 1).sum())

    # Dedup
    if args.dedup_policy == "best":
        merged = dedup_best(combined)
    else:
        merged = dedup_prefer_last(combined)

    # Ensure exact column order
    merged = merged[COL_ORDER]

    # Gaps
    ids = merged["simulation_id"].dropna().astype(int).unique()
    ids.sort()
    gap_ranges = find_gap_ranges(ids)

    min_id = int(ids.min()) if len(ids) else None
    max_id = int(ids.max()) if len(ids) else None

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(str(out_path), index=False)

    # Report
    print("=== Merge Summary ===")
    for p, d in zip(inputs, dfs):
        print(f"Input: {p}  rows={len(d):,}")
    print(f"Combined rows (pre-dedup) = {len(combined):,}")
    print(f"Duplicate simulation_id count (IDs) = {num_dup_ids:,}")
    print(f"Extra duplicate rows (beyond first per ID) = {extra_dup_rows:,}")
    print(f"Final merged rows = {len(merged):,}")
    print(f"simulation_id range = {min_id} .. {max_id}")
    print(f"Saved: {out_path}")

    print("\n=== Gap Report (missing simulation_id ranges) ===")
    if not gap_ranges:
        print("No gaps detected (continuous IDs).")
    else:
        to_show = gap_ranges[: max(0, args.show_gaps)]
        for (s, e) in to_show:
            if s == e:
                print(f"Missing: {s}")
            else:
                print(f"Missing: {s}–{e}  (count={e - s + 1})")
        if len(gap_ranges) > args.show_gaps:
            print(f"... and {len(gap_ranges) - args.show_gaps} more gap ranges.")

    if args.expected_gap is not None:
        gs, ge = args.expected_gap
        has_expected_gap = any(s <= gs and e >= ge for s, e in gap_ranges)
        print("\n=== Expected gap check ===")
        if has_expected_gap:
            print(f"Yes: a gap covering {gs}–{ge} exists in the merged results.")
        else:
            print(f"No: the merged results do NOT contain a full gap covering {gs}–{ge}.")

    # Non-zero exit if no rows written (optional safety)
    if len(merged) == 0:
        print("WARNING: Output is empty after filtering/dedup.", file=sys.stderr)


if __name__ == "__main__":
    main()
