# canopy_data_generation_surrogate_pipeline

Pipeline for generating maize canopy architectures, running Helios PAR simulations, and building datasets for surrogate model training.

The workflow starts from sampled canopy parameters, generates canopy geometry, runs `par_parallel` to compute `net_PAR`, and prepares ML-ready datasets for downstream surrogate modeling.

## Repository structure

```text
canopy_data_generation_surrogate_pipeline/
├── src/
│   └── maize_canopy_pipeline/
│       ├── data_generation/
│       └── ml/
├── scripts/
│   ├── run_classical_training.py
│   ├── run_nn_training.py
│   └── slurm/
├── tools/
├── data/
├── environment.yml
├── requirements.txt
└── README.md
```

- `src/maize_canopy_pipeline/data_generation/` — canopy generation, simulation, and dataset building
- `src/maize_canopy_pipeline/ml/` — classical and neural surrogate models
- `scripts/` — entry points for training and cluster runs

## Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate <env-name>
```

Or install with pip:

```bash
pip install -r requirements.txt
```

Main dependencies include `geomdl`, `open3d`, `scikit-learn`, and `torch`.

## Data generation workflow

1. Generate a Sobol catalog of canopy parameters  
2. Convert parameters to canopy geometry and export OBJ files  
3. Run Helios `par_parallel` to compute PAR  
4. Build a scaled catalog of canopy traits  
5. Merge inputs with `net_PAR` outputs to create the final ML dataset  

## ML models

This repository supports two training workflows:

- **Classical / boosted models**: linear models, random forests, gradient boosting, XGBoost, LightGBM, CatBoost
- **Neural models**: DeepSets, Leaf Transformer, ResNet-MLP, Mixture-of-Experts MLP

Run classical models:

```bash
python scripts/run_classical_training.py --data_csv results/ml_dataset_scaled.csv --out_dir results/classical_model_zoo
```

Run neural models:

```bash
python scripts/run_nn_training.py --data_csv results/ml_dataset_scaled.csv --out_dir results/nn_model_zoo
```

## Helios dependency

This pipeline depends on a separate local build of Helios. The `par_parallel` executable is not included in this repository.

## Notes

- Large generated datasets and simulation outputs are not tracked in the repo
- Helios binaries must be provided separately
- This repo focuses on reproducible pipeline code and surrogate training
