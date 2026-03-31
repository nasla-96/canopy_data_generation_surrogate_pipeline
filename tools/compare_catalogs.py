#!/usr/bin/env python3
import hashlib
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd


FILE1 = "../catalog/sim_catalog_scaled.csv"
FILE2 = "../catalog/sim_catalog_scaled_2.csv"
ID_COL = "simulation_id"

# If you want exact equality, keep FLOAT_TOL=None.
# If your files might differ only by tiny float formatting, use something like 1e-12 or 1e-10.
FLOAT_TOL = 1e-12

# Quantize for robust hashing when FLOAT_TOL is used:
# with tol=1e-12, rounding to 12 decimals is reasonable.
ROUND_DECIMALS = 12 if FLOAT_TOL is not None else None


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def row_fingerprint(values: np.ndarray) -> str:
    """
    Create a stable fingerprint for a 1D numpy array of mixed types.
    Uses SHA1 of a byte representation that is consistent across runs.
    """
    # Convert to bytes via pandas to_string-ish stable join
    # This avoids issues with numpy object dtype bytes.
    s = "|".join("" if pd.isna(x) else str(x) for x in values.tolist())
    return sha1_bytes(s.encode("utf-8"))


def vector_fingerprints(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """
    Return a Series indexed like df, with fingerprint per row for cols.
    """
    arr = df[cols].to_numpy(dtype=object)
    fps = [row_fingerprint(arr[i, :]) for i in range(arr.shape[0])]
    return pd.Series(fps, index=df.index, name="fp")


def canonicalize_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    If FLOAT_TOL is enabled, round numeric columns to ROUND_DECIMALS so that hashing
    is tolerant to tiny float formatting differences.
    Non-numeric columns are left as-is.
    """
    out = df[cols].copy()
    if FLOAT_TOL is None:
        return out

    for c in cols:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").round(ROUND_DECIMALS)
        else:
            # normalize missing
            out[c] = out[c].astype("object")
    return out


def main() -> int:
    print("=== LOADING ===")
    df1 = pd.read_csv(FILE1)
    df2 = pd.read_csv(FILE2)
    print("file1:", FILE1, "shape:", df1.shape)
    print("file2:", FILE2, "shape:", df2.shape)
    print()

    if ID_COL not in df1.columns or ID_COL not in df2.columns:
        print(f"ERROR: '{ID_COL}' must exist in both files.")
        return 2

    # Column sets
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    common_cols = sorted((cols1 & cols2) - {ID_COL})
    only1 = sorted(cols1 - cols2)
    only2 = sorted(cols2 - cols1)

    print("=== COLUMNS ===")
    print("common (excluding id):", len(common_cols))
    if only1:
        print("only in file1:", only1)
    if only2:
        print("only in file2:", only2)
    if only1 or only2:
        print("ERROR: Column sets differ; cannot compare sampling reliably.")
        return 2
    print()

    # ID integrity
    print("=== ID INTEGRITY ===")
    dup1 = df1[ID_COL].duplicated().sum()
    dup2 = df2[ID_COL].duplicated().sum()
    print("duplicate IDs file1:", dup1)
    print("duplicate IDs file2:", dup2)
    if dup1 or dup2:
        print("ERROR: duplicate simulation_id exists; fix duplicates first.")
        return 2

    df1 = df1.set_index(ID_COL).sort_index()
    df2 = df2.set_index(ID_COL).sort_index()

    ids1 = df1.index
    ids2 = df2.index
    missing_in_2 = ids1.difference(ids2)
    missing_in_1 = ids2.difference(ids1)
    print("n_ids file1:", len(ids1))
    print("n_ids file2:", len(ids2))
    print("missing in file2:", len(missing_in_2))
    print("missing in file1:", len(missing_in_1))
    if len(missing_in_2) or len(missing_in_1):
        print("ERROR: ID sets differ; cannot do strict by-ID comparison.")
        # Still could check sampling-set ignoring IDs below, but that's likely not what you want.
    print()

    # --- Check A: same vectors for same simulation_id ---
    print("=== CHECK A: BY-ID EQUALITY (same sim_id => same parameter vector) ===")
    common_ids = ids1.intersection(ids2)
    a = df1.loc[common_ids, common_cols]
    b = df2.loc[common_ids, common_cols]

    if FLOAT_TOL is None:
        by_id_equal_mask = (a.astype("object").fillna("<NA>") == b.astype("object").fillna("<NA>")).all(axis=1)
    else:
        # numeric tol + exact non-numeric
        num_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(a[c])]
        non_cols = [c for c in common_cols if c not in num_cols]

        an = a[num_cols].to_numpy()
        bn = b[num_cols].to_numpy()
        num_ok = np.isclose(an, bn, atol=FLOAT_TOL, rtol=0.0, equal_nan=True).all(axis=1)

        if non_cols:
            non_ok = (a[non_cols].astype("object").fillna("<NA>") ==
                      b[non_cols].astype("object").fillna("<NA>")).all(axis=1).to_numpy()
        else:
            non_ok = np.ones(len(common_ids), dtype=bool)

        by_id_equal_mask = pd.Series(num_ok & non_ok, index=common_ids)

    n_match_by_id = int(by_id_equal_mask.sum())
    n_total = len(common_ids)
    n_bad = n_total - n_match_by_id
    print(f"matched by same sim_id: {n_match_by_id}/{n_total}")
    if n_bad == 0:
        print("✅ Files are IDENTICAL by simulation_id (same sampling and same ID->sample mapping).")
        print()
    else:
        bad_ids = by_id_equal_mask.index[~by_id_equal_mask].tolist()
        print(f"❌ NOT identical by simulation_id. Differing sim_id count: {n_bad}")
        print("first 10 differing sim_id:", bad_ids[:10])

        # Show a compact diff for 1 example ID
        ex = bad_ids[0]
        diff_cols = []
        if FLOAT_TOL is None:
            neq = (a.loc[ex].astype("object").fillna("<NA>") != b.loc[ex].astype("object").fillna("<NA>"))
            diff_cols = neq[neq].index.tolist()
        else:
            # Compute per-col mismatch for example
            row_a = a.loc[ex]
            row_b = b.loc[ex]
            for c in common_cols:
                va, vb = row_a[c], row_b[c]
                if pd.isna(va) and pd.isna(vb):
                    continue
                if pd.api.types.is_numeric_dtype(a[c]):
                    if not np.isclose(float(va), float(vb), atol=FLOAT_TOL, rtol=0.0, equal_nan=True):
                        diff_cols.append(c)
                else:
                    if (pd.isna(va) != pd.isna(vb)) or (str(va) != str(vb)):
                        diff_cols.append(c)

        print(f"example sim_id={ex} differs in {len(diff_cols)} columns (showing up to 15):")
        print(diff_cols[:15])
        if diff_cols:
            show = diff_cols[:8]
            print("\nExample values (file1 vs file2) for a few differing cols:")
            for c in show:
                print(f"  {c}: {a.loc[ex, c]}  |  {b.loc[ex, c]}")
        print()

    # --- Check B: same sampling set ignoring IDs (multiset of vectors) ---
    # This answers: are these the same Sobol points but permuted / re-labeled?
    print("=== CHECK B: SAMPLING-SET EQUIVALENCE (ignore simulation_id) ===")
    # Canonicalize values (round numerics if tol enabled) to reduce false negatives
    a_can = canonicalize_numeric(df1.reset_index(), common_cols)
    b_can = canonicalize_numeric(df2.reset_index(), common_cols)

    # Fingerprints per row
    fp1 = vector_fingerprints(a_can, common_cols)
    fp2 = vector_fingerprints(b_can, common_cols)

    vc1 = fp1.value_counts()
    vc2 = fp2.value_counts()

    # Compare multisets via counts
    # (This handles duplicates, though Sobol should have essentially none.)
    all_keys = vc1.index.union(vc2.index)
    count_diff = (vc1.reindex(all_keys, fill_value=0) - vc2.reindex(all_keys, fill_value=0))
    n_mismatch_keys = int((count_diff != 0).sum())

    if n_mismatch_keys == 0:
        print("✅ Same sampling set (same Sobol points) up to permutation / ID mapping.")
        if n_bad != 0:
            print("   Interpretation: samples are the same set, but simulation_id->sample mapping differs.")
    else:
        print("❌ Different sampling set (Sobol points differ).")
        # quantify how different
        # keys with positive diff exist more in file1; negative more in file2
        extra_in_1 = int(count_diff[count_diff > 0].sum())
        extra_in_2 = int((-count_diff[count_diff < 0]).sum())
        print("rows present more often in file1 (total count):", extra_in_1)
        print("rows present more often in file2 (total count):", extra_in_2)

        # show a few example fingerprints unique to either file
        uniq1 = count_diff[count_diff > 0].index[:5].tolist()
        uniq2 = count_diff[count_diff < 0].index[:5].tolist()
        print("example unique fingerprints (file1):", uniq1)
        print("example unique fingerprints (file2):", uniq2)

        # show one example actual row from each side (helpful for debugging)
        if uniq1:
            exfp = uniq1[0]
            exrow = a_can.loc[fp1[fp1 == exfp].index[0], common_cols]
            print("\nExample row present in file1 but not file2 (first match):")
            print(exrow.to_string())
        if uniq2:
            exfp = uniq2[0]
            exrow = b_can.loc[fp2[fp2 == exfp].index[0], common_cols]
            print("\nExample row present in file2 but not file1 (first match):")
            print(exrow.to_string())

    print()
    print("=== SUMMARY ===")
    if n_bad == 0 and n_mismatch_keys == 0:
        print("✅ Same sampling and same sim_id mapping (identical catalogs).")
    elif n_bad != 0 and n_mismatch_keys == 0:
        print("⚠️ Same sampling set, but sim_id->sample mapping differs (permuted/re-labeled IDs).")
    else:
        print("❌ Different sampling (Sobol points differ).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())