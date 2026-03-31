# Scripts

Main command-line entry points for the canopy data-generation pipeline.

- `generate_catalog.py`: create raw Sobol catalogs
- `generate_canopies.py`: convert catalog rows into canopy OBJ files
- `run_helios_single.py`: run one Helios PAR simulation
- `run_pipeline.py`: batch wrapper for catalog -> canopy -> Helios
- `make_scaled_catalog.py`: convert raw catalog values to physically scaled features
- `build_ml_dataset.py`: merge scaled inputs with Helios PAR outputs

`slurm/` contains cluster launch scripts.