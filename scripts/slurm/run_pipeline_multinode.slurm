#!/bin/bash
#SBATCH --job-name=canopy_par
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/%x_%j_%N_%t.out
#SBATCH --error=logs/%x_%j_%N_%t.err

set -euo pipefail

# -------------------------
# Fill these paths
# -------------------------
REPO=/work/mech-ai-scratch/nasla/canopy_data_generation_surrogate_pipeline
HELIOS_BUILD_DIR=/work/mech-ai-scratch/nasla/helios_new/Helios/samples/par_parallel/build
BASE_JSON=/work/mech-ai-scratch/nasla/path/to/your_base_plant.json
INI_TEMPLATE=/work/mech-ai-scratch/nasla/path/to/base_par_config.ini

# -------------------------
# Run settings
# -------------------------
CATALOG=${REPO}/catalog/sim_catalog.csv
CATALOG_N_SIMS=10000          # only used if catalog does not already exist
START_ID=0
END_ID=10000

RUN_TAG=run_${SLURM_JOB_ID}
CANOPY_DIR=${REPO}/canopies/${RUN_TAG}
CONFIG_DIR=${REPO}/configs/${RUN_TAG}
RESULTS_DIR=${REPO}/results/${RUN_TAG}
MERGED_RESULTS=${RESULTS_DIR}/results_merged.csv
SCALED_CATALOG=${RESULTS_DIR}/sim_catalog_scaled.csv
ML_DATASET=${RESULTS_DIR}/ml_dataset.csv

mkdir -p "${REPO}/logs" "${REPO}/catalog" "${CANOPY_DIR}" "${CONFIG_DIR}" "${RESULTS_DIR}"

source ~/.bashrc
conda activate nurbs
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

# Optional: load runtime modules here if Helios needs them on Nova
# module purge
# module load mesa-glu
# module load boost/1.86.0-gvbpdbb

if [ ! -x "${HELIOS_BUILD_DIR}/par_parallel" ]; then
    echo "ERROR: par_parallel not found at ${HELIOS_BUILD_DIR}/par_parallel"
    exit 1
fi

# Create catalog once if missing
if [ ! -s "${CATALOG}" ]; then
    python "${REPO}/scripts/generate_catalog.py" \
        --n_sims "${CATALOG_N_SIMS}" \
        --out_path "${CATALOG}" \
        --lon_deg 93.62
fi

# -------------------------
# One task per GPU
# Each task gets a unique shard of simulation IDs
# -------------------------
srun --kill-on-bad-exit=1 bash -lc '
set -euo pipefail

RANK=${SLURM_PROCID}
LOCAL_GPU=${SLURM_LOCALID}
WORLD_SIZE=${SLURM_NTASKS}

BASE_START='"${START_ID}"'
BASE_END='"${END_ID}"'
TOTAL=$((BASE_END - BASE_START))
CHUNK=$(((TOTAL + WORLD_SIZE - 1) / WORLD_SIZE))

MY_START=$((BASE_START + RANK * CHUNK))
MY_END=$((MY_START + CHUNK))

if [ "${MY_END}" -gt "${BASE_END}" ]; then
    MY_END=${BASE_END}
fi

if [ "${MY_START}" -ge "${BASE_END}" ]; then
    echo "Rank ${RANK}: no work"
    exit 0
fi

echo "Host=$(hostname) rank=${RANK} local_gpu=${LOCAL_GPU} range=${MY_START}:${MY_END}"

python '"${REPO}"'/scripts/run_pipeline.py \
    --start_id "${MY_START}" \
    --end_id "${MY_END}" \
    --catalog '"${CATALOG}"' \
    --base_json '"${BASE_JSON}"' \
    --canopy_dir '"${CANOPY_DIR}"' \
    --ini_template '"${INI_TEMPLATE}"' \
    --config_out_dir '"${CONFIG_DIR}"' \
    --helios_build_dir '"${HELIOS_BUILD_DIR}"' \
    --device_ids "${LOCAL_GPU}" \
    --results_csv '"${RESULTS_DIR}"'/results_rank_${RANK}.csv \
    --skip_existing_obj \
    --skip_existing_results \
    --no_modules
'

# -------------------------
# Merge shard outputs once
# -------------------------
python "${REPO}/tools/merge_results.py" \
    --inputs "${RESULTS_DIR}"/results_rank_*.csv \
    --out "${MERGED_RESULTS}" \
    --dedup-policy best

# -------------------------
# Build ML artifacts once
# -------------------------
python "${REPO}/scripts/make_scaled_catalog.py" \
    --in_catalog "${CATALOG}" \
    --out_catalog "${SCALED_CATALOG}" \
    --base_json "${BASE_JSON}"

python "${REPO}/scripts/build_ml_dataset.py" \
    --scaled_catalog "${SCALED_CATALOG}" \
    --results "${MERGED_RESULTS}" \
    --out "${ML_DATASET}"

echo "Done."
echo "Merged results: ${MERGED_RESULTS}"
echo "Scaled catalog: ${SCALED_CATALOG}"
echo "ML dataset: ${ML_DATASET}"