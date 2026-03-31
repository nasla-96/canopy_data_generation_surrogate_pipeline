import pandas as pd
import glob
import os

# Folder containing CSV files
folder = "../results"   # change if needed

# Find all CSV files
csv_files = glob.glob(os.path.join(folder, "*.csv"))

print(f"Found {len(csv_files)} CSV files")

dfs = []

for f in csv_files:
    print("Reading:", f)
    df = pd.read_csv(f)
    dfs.append(df)

# Merge all
merged = pd.concat(dfs, ignore_index=True)

print("Rows before dedup:", len(merged))

# Remove duplicate simulation_ids if any
merged = merged.drop_duplicates(subset="simulation_id", keep="first")

print("Rows after dedup:", len(merged))

# Sort by simulation_id
merged = merged.sort_values("simulation_id").reset_index(drop=True)

# Save merged CSV
out_path = os.path.join(folder, "merged_results.csv")
merged.to_csv(out_path, index=False)

print("Saved merged file:", out_path)
print("Final rows:", len(merged))
print("Simulation ID range:", merged["simulation_id"].min(), "to", merged["simulation_id"].max())