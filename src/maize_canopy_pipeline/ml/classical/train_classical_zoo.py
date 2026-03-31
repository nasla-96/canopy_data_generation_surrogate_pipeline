#!/usr/bin/env python3
"""
Classical + boosted model zoo for Helios surrogate (tabular regression).

v2+ (cluster-friendly rewrite):
- Adds --models / --skip_models to run subsets (for multi-node splitting).
- Adds --cv, --n_iter, --n_jobs to control search cost + CPU parallelism.
- Keeps v2 features: canopy-derived deterministic shading proxies + robust geo encodings.
- Adds explicit XGBoost runtime toggles (tree_method + device).

Example (single model, node-splitting):
  python fit_model_zoo_v2.py \
    --data_csv results/ml_dataset_scaled.csv \
    --models xgboost \
    --out_dir results/fit_xgb \
    --geo_mode raw+ecef+sincos \
    --add_canopy_feats \
    --cv 5 \
    --n_iter 60 \
    --n_jobs 12

Example (run just your recommended classical set):
  python fit_model_zoo_v2.py \
    --data_csv results/ml_dataset_scaled.csv \
    --models random_forest extra_trees_big gbrt hist_gbdt xgboost lightgbm catboost \
    --out_dir results/model_zoo_reco \
    --geo_mode raw+ecef+sincos \
    --add_canopy_feats \
    --cv 5 \
    --n_iter 60 \
    --n_jobs 32
"""

import os
import json
import joblib
import warnings
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Linear / robust
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, SGDRegressor, LassoCV
)

# Neighbors / kernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

# Trees / ensembles
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# Optional: neural net baseline
from sklearn.neural_network import MLPRegressor


RANDOM_SEED = 42
TARGET_COL = "net_PAR"

# Bookkeeping columns to drop if present
NON_FEATURE_COLS = [
    "simulation_id", "status", "error", "obj_path", "config_path", "device_id", "stage",
]

# Optional external boosters
HAVE_XGB = HAVE_LGBM = HAVE_CAT = False
try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    pass

try:
    from lightgbm import LGBMRegressor
    HAVE_LGBM = True
except Exception:
    pass

try:
    from catboost import CatBoostRegressor
    HAVE_CAT = True
except Exception:
    pass


def evaluate(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def make_preprocessors(numeric_cols):
    scaled_num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    scaled_pre = ColumnTransformer([("num", scaled_num, numeric_cols)], remainder="drop")

    tree_num = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    tree_pre = ColumnTransformer([("num", tree_num, numeric_cols)], remainder="drop")
    return scaled_pre, tree_pre


def _ecef_from_latlon(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """
    Unit-sphere ECEF encoding: (x,y,z) from lat/lon.
    Keeps lon meaningful and removes the +/-180 discontinuity.
    """
    lat = np.deg2rad(lat_deg.astype(float))
    lon = np.deg2rad(lon_deg.astype(float))
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=1)


def add_geo_features(df: pd.DataFrame, mode: str = "raw+ecef") -> pd.DataFrame:
    """
    Keep raw lat/lon, optionally add ecef and/or sin/cos(lon).
    """
    if ("lat_deg" not in df.columns) or ("lon_deg" not in df.columns):
        return df

    out = df.copy()
    lat = out["lat_deg"].to_numpy()
    lon = out["lon_deg"].to_numpy()

    if "ecef" in mode:
        ecef = _ecef_from_latlon(lat, lon)
        out["geo_x"] = ecef[:, 0]
        out["geo_y"] = ecef[:, 1]
        out["geo_z"] = ecef[:, 2]

    if "sincos" in mode:
        lonr = np.deg2rad(lon.astype(float))
        out["lon_sin"] = np.sin(lonr)
        out["lon_cos"] = np.cos(lonr)

    return out


def add_canopy_derived_features(df: pd.DataFrame, n_leaves: int = 12) -> pd.DataFrame:
    """
    Deterministic engineered features to make shading/interaction effects easier to learn
    WITHOUT needing per-layer PAR labels.

    Adds per-leaf (sorted by interleaf_pos within row):
      - area_sorted_j = length*width
      - proj_sorted_j = area*abs(cos(theta))  (proxy for shading strength)
      - hnorm_sorted_j in [0,1] from interleaf_pos across leaves
      - rank_sorted_j in [0,1] by height order
      - cum_area_above_sorted_j, cum_proj_above_sorted_j

    Adds canopy summaries:
      - canopy_area_sum, canopy_proj_sum, canopy_hspread, canopy_area_top_frac, canopy_area_bottom_frac
    """
    out = df.copy()

    def col(name, i): return f"{name}_{i}"

    req = []
    for i in range(n_leaves):
        req += [col("interleaf_pos", i), col("length_cm", i), col("width_cm", i), col("theta_deg", i)]
    missing = [c for c in req if c not in out.columns]
    if missing:
        # silent no-op if schema doesn't match expected
        return out

    pos = np.stack([out[col("interleaf_pos", i)].to_numpy(dtype=float) for i in range(n_leaves)], axis=1)
    length = np.stack([out[col("length_cm", i)].to_numpy(dtype=float) for i in range(n_leaves)], axis=1)
    width = np.stack([out[col("width_cm", i)].to_numpy(dtype=float) for i in range(n_leaves)], axis=1)
    theta = np.stack([out[col("theta_deg", i)].to_numpy(dtype=float) for i in range(n_leaves)], axis=1)

    area = length * width
    proj = area * np.abs(np.cos(np.deg2rad(theta)))

    order = np.argsort(pos, axis=1)  # bottom -> top
    pos_s = np.take_along_axis(pos, order, axis=1)
    area_s = np.take_along_axis(area, order, axis=1)
    proj_s = np.take_along_axis(proj, order, axis=1)

    eps = 1e-9
    hmin = pos_s.min(axis=1, keepdims=True)
    hmax = pos_s.max(axis=1, keepdims=True)
    hnorm = (pos_s - hmin) / (hmax - hmin + eps)

    rank = np.linspace(0.0, 1.0, n_leaves, dtype=float)[None, :].repeat(pos.shape[0], axis=0)

    suffix_area = np.cumsum(area_s[:, ::-1], axis=1)[:, ::-1]
    suffix_proj = np.cumsum(proj_s[:, ::-1], axis=1)[:, ::-1]
    cum_area_above = suffix_area - area_s
    cum_proj_above = suffix_proj - proj_s

    for j in range(n_leaves):
        out[f"area_sorted_{j}"] = area_s[:, j]
        out[f"proj_sorted_{j}"] = proj_s[:, j]
        out[f"hnorm_sorted_{j}"] = hnorm[:, j]
        out[f"rank_sorted_{j}"] = rank[:, j]
        out[f"cum_area_above_sorted_{j}"] = cum_area_above[:, j]
        out[f"cum_proj_above_sorted_{j}"] = cum_proj_above[:, j]

    out["canopy_area_sum"] = area_s.sum(axis=1)
    out["canopy_proj_sum"] = proj_s.sum(axis=1)
    out["canopy_hspread"] = (hmax - hmin).reshape(-1)
    out["canopy_area_top_frac"] = area_s[:, -4:].sum(axis=1) / (out["canopy_area_sum"].to_numpy() + eps)
    out["canopy_area_bottom_frac"] = area_s[:, :4].sum(axis=1) / (out["canopy_area_sum"].to_numpy() + eps)
    return out


def build_zoo(scaled_pre, tree_pre, n_jobs_models: int, args) -> list:
    """
    Returns list of (name, pipeline, param_space).
    n_jobs_models is used for estimators that accept n_jobs.
    args contains XGB/LGBM runtime toggles.
    """
    zoo = []

    # Linear baselines
    zoo.append(("linear", Pipeline([("pre", scaled_pre), ("model", LinearRegression())]), {}))
    zoo.append(("ridge", Pipeline([("pre", scaled_pre), ("model", Ridge())]),
                {"model__alpha": np.logspace(-3, 4, 50)}))
    zoo.append(("lasso", Pipeline([("pre", scaled_pre), ("model", Lasso(max_iter=200000))]),
                {"model__alpha": np.logspace(-5, 1, 60)}))
    zoo.append(("elasticnet", Pipeline([("pre", scaled_pre), ("model", ElasticNet(max_iter=200000))]),
                {"model__alpha": np.logspace(-5, 1, 60), "model__l1_ratio": np.linspace(0.05, 0.95, 10)}))
    zoo.append(("huber", Pipeline([("pre", scaled_pre), ("model", HuberRegressor(max_iter=5000))]),
                {"model__epsilon": np.linspace(1.1, 2.5, 15), "model__alpha": np.logspace(-6, -1, 20)}))
    zoo.append(("sgd", Pipeline([("pre", scaled_pre), ("model", SGDRegressor(random_state=RANDOM_SEED))]),
                {"model__loss": ["squared_error", "huber"],
                 "model__alpha": np.logspace(-6, -2, 30),
                 "model__penalty": ["l2", "l1", "elasticnet"],
                 "model__learning_rate": ["invscaling", "adaptive"],
                 "model__eta0": np.logspace(-4, -1, 10),
                 "model__max_iter": [5000]}))

    # Interpretable interactions
    zoo.append(("poly2_ridge",
                Pipeline([("pre", scaled_pre),
                          ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                          ("model", Ridge())]),
                {"model__alpha": np.logspace(-4, 4, 30)}))

    zoo.append(("sindy_poly2_lassocv",
                Pipeline([("pre", scaled_pre),
                          ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                          ("model", LassoCV(cv=5, n_alphas=80, max_iter=200000, random_state=RANDOM_SEED))]),
                {}))

    # kNN / kernel
    zoo.append(("knn", Pipeline([("pre", scaled_pre), ("model", KNeighborsRegressor())]),
                {"model__n_neighbors": list(range(3, 61, 2)),
                 "model__weights": ["uniform", "distance"],
                 "model__p": [1, 2]}))

    zoo.append(("svr_rbf", Pipeline([("pre", scaled_pre), ("model", SVR(kernel="rbf"))]),
                {"model__C": np.logspace(-1, 3, 20),
                 "model__gamma": np.logspace(-6, -1, 20),
                 "model__epsilon": np.logspace(-3, 0, 10)}))

    zoo.append(("krr_rbf", Pipeline([("pre", scaled_pre), ("model", KernelRidge(kernel="rbf"))]),
                {"model__alpha": np.logspace(-3, 2, 20),
                 "model__gamma": np.logspace(-6, -1, 20)}))

    # Tree ensembles
    zoo.append(("random_forest",
                Pipeline([("pre", tree_pre),
                          ("model", RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=n_jobs_models))]),
                {"model__n_estimators": [600, 1000, 2000],
                 "model__max_depth": [12, 16, 20, None],
                 "model__min_samples_leaf": [2, 5, 10, 20],
                 "model__max_features": ["sqrt", 0.3, 0.5, 0.8]}))

    zoo.append(("extra_trees_big",
                Pipeline([("pre", tree_pre),
                          ("model", ExtraTreesRegressor(random_state=RANDOM_SEED, n_jobs=n_jobs_models))]),
                {"model__n_estimators": [1000, 2000, 4000],
                 "model__max_depth": [16, 20, None],
                 "model__min_samples_leaf": [2, 5, 10, 20],
                 "model__max_features": ["sqrt", 0.3, 0.5, 0.8]}))

    zoo.append(("gbrt",
                Pipeline([("pre", tree_pre), ("model", GradientBoostingRegressor(random_state=RANDOM_SEED))]),
                {"model__n_estimators": [600, 1000, 1500],
                 "model__learning_rate": np.logspace(-2.8, -0.5, 14),
                 "model__max_depth": [2, 3, 4],
                 "model__subsample": [0.6, 0.8, 1.0]}))

    zoo.append(("hist_gbdt",
                Pipeline([("pre", tree_pre), ("model", HistGradientBoostingRegressor(random_state=RANDOM_SEED))]),
                {"model__learning_rate": np.logspace(-2.8, -0.5, 14),
                 "model__max_depth": [3, 5, 7, None],
                 "model__max_leaf_nodes": [31, 63, 127, 255],
                 "model__min_samples_leaf": [10, 20, 40, 80],
                 "model__l2_regularization": np.logspace(-7, 0, 12)}))

    # MLP baseline
    zoo.append(("mlp",
                Pipeline([("pre", scaled_pre),
                          ("model", MLPRegressor(
                              random_state=RANDOM_SEED,
                              early_stopping=True,
                              validation_fraction=0.15,
                              n_iter_no_change=25,
                              max_iter=6000,
                          ))]),
                {"model__hidden_layer_sizes": [(128,), (256,), (128, 64), (256, 128), (512, 256)],
                 "model__activation": ["relu", "tanh"],
                 "model__alpha": np.logspace(-5, 1, 20),
                 "model__learning_rate_init": np.logspace(-4, -2, 10)}))

    # Boosters
    if HAVE_XGB:
        # XGB runtime toggles:
        # - tree_method: hist (CPU), gpu_hist (GPU)
        # - device: cpu/cuda (newer XGBoost)
        xgb_kwargs = dict(
            random_state=RANDOM_SEED,
            n_estimators=6000,
            tree_method=args.xgb_tree_method,
            objective="reg:squarederror",
            n_jobs=n_jobs_models,
        )
        # device is supported in newer versions; safe to pass if available
        if args.xgb_device is not None:
            xgb_kwargs["device"] = args.xgb_device

        zoo.append(("xgboost",
                    Pipeline([("pre", tree_pre), ("model", XGBRegressor(**xgb_kwargs))]),
                    {"model__max_depth": [3, 4, 5, 6, 8],
                     "model__learning_rate": np.logspace(-3.2, -0.8, 14),
                     "model__subsample": [0.6, 0.8, 1.0],
                     "model__colsample_bytree": [0.4, 0.6, 0.8, 1.0],
                     "model__min_child_weight": [1, 5, 10, 20],
                     "model__reg_alpha": np.logspace(-8, -1, 10),
                     "model__reg_lambda": np.logspace(-3, 1, 10)}))

    if HAVE_LGBM:
        # Note: LightGBM GPU requires a GPU-enabled build; pip wheels are usually CPU-only.
        lgbm_kwargs = dict(
            random_state=RANDOM_SEED,
            n_estimators=12000,
            n_jobs=n_jobs_models,
        )
        zoo.append(("lightgbm",
                    Pipeline([("pre", tree_pre), ("model", LGBMRegressor(**lgbm_kwargs))]),
                    {"model__learning_rate": np.logspace(-3.2, -0.8, 14),
                     "model__num_leaves": [31, 63, 127, 255, 511],
                     "model__max_depth": [-1, 6, 10, 14, 18],
                     "model__min_child_samples": [10, 20, 40, 80, 120],
                     "model__subsample": [0.6, 0.8, 1.0],
                     "model__colsample_bytree": [0.5, 0.7, 0.9, 1.0],
                     "model__reg_alpha": np.logspace(-8, -1, 10),
                     "model__reg_lambda": np.logspace(-3, 1, 10)}))

    if HAVE_CAT:
        zoo.append(("catboost",
                    Pipeline([("pre", tree_pre),
                              ("model", CatBoostRegressor(
                                  random_seed=RANDOM_SEED,
                                  verbose=False,
                                  loss_function="RMSE",
                              ))]),
                    {"model__depth": [4, 6, 8, 10],
                     "model__learning_rate": np.logspace(-3.2, -0.8, 14),
                     "model__l2_leaf_reg": np.logspace(-3, 2, 10),
                     "model__iterations": [4000, 8000, 12000]}))

    return zoo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="results/ml_dataset_scaled.csv")
    ap.add_argument("--target", default=TARGET_COL)
    ap.add_argument("--out_dir", default="ML_models/results/model_zoo_v2")
    ap.add_argument("--geo_mode", default="raw+ecef", choices=["raw", "raw+ecef", "raw+sincos", "raw+ecef+sincos"])
    ap.add_argument("--add_canopy_feats", action="store_true", help="Add canopy-derived features (recommended)")
    ap.add_argument("--n_leaves", type=int, default=12)

    # NEW: subset control
    ap.add_argument("--models", nargs="+", default=None,
                    help="Optional list of model names to run (space-separated). If omitted, runs all.")
    ap.add_argument("--skip_models", nargs="+", default=None,
                    help="Optional list of model names to skip.")

    # NEW: search controls
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=60)
    ap.add_argument("--n_jobs", type=int, default=-1,
                    help="Parallel jobs for RandomizedSearchCV and n_jobs-capable estimators. "
                         "Use a positive integer per process on multi-tenant nodes.")

    # NEW: XGBoost runtime toggles
    ap.add_argument("--xgb_tree_method", default="hist", choices=["hist", "gpu_hist"],
                    help="XGBoost tree_method. Use gpu_hist only if you want GPU XGBoost.")
    ap.add_argument("--xgb_device", default=None, choices=[None, "cpu", "cuda"],
                    help="XGBoost device (newer XGBoost). Typically 'cuda' when using gpu_hist.")

    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data_csv)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in {args.data_csv}")

    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    df = df.loc[np.isfinite(df[args.target])].copy()

    drop_cols = [c for c in NON_FEATURE_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    df = add_geo_features(df, mode=args.geo_mode)
    if args.add_canopy_feats:
        df = add_canopy_derived_features(df, n_leaves=args.n_leaves)

    X = df.drop(columns=[args.target])
    y = df[args.target].astype(float)

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found after cleaning.")

    X = X[numeric_cols]

    print(f"Loaded: {args.data_csv}")
    print(f"Rows: {len(df)}")
    print(f"Using {len(numeric_cols)} numeric feature columns")
    print("First 20 features:", numeric_cols[:20])
    print(f"cv={args.cv} n_iter={args.n_iter} n_jobs={args.n_jobs}")
    print(f"geo_mode={args.geo_mode} add_canopy_feats={bool(args.add_canopy_feats)}")

    # # Split
    # X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_SEED)
    # X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_SEED)

    # X_trval = pd.concat([X_train, X_val], axis=0)
    # y_trval = pd.concat([y_train, y_val], axis=0)

    # Ordered Sobol split (first 10% test, next 10% val, rest train)

    n = len(X)

    n_test = int(0.10 * n)
    n_val  = int(0.10 * n)

    # slices
    test_idx  = slice(0, n_test)
    val_idx   = slice(n_test, n_test + n_val)
    train_idx = slice(n_test + n_val, n)

    X_test  = X.iloc[test_idx].copy()
    y_test  = y.iloc[test_idx].copy()

    X_val   = X.iloc[val_idx].copy()
    y_val   = y.iloc[val_idx].copy()

    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()

    # this keeps compatibility with rest of script
    X_trval = pd.concat([X_train, X_val], axis=0)
    y_trval = pd.concat([y_train, y_val], axis=0)

    print("Ordered Sobol split:")
    print(f"Test rows: 0 → {n_test}")
    print(f"Val rows: {n_test} → {n_test+n_val}")
    print(f"Train rows: {n_test+n_val} → {n}")
    scaled_pre, tree_pre = make_preprocessors(numeric_cols)

    # Build zoo
    zoo = build_zoo(scaled_pre, tree_pre, n_jobs_models=args.n_jobs, args=args)

    all_names = [name for name, _, _ in zoo]
    if args.models is not None:
        unknown = [m for m in args.models if m not in all_names]
        if unknown:
            raise ValueError(f"Unknown model(s) in --models: {unknown}. Available: {all_names}")
        zoo = [z for z in zoo if z[0] in set(args.models)]

    if args.skip_models is not None:
        unknown = [m for m in args.skip_models if m not in all_names]
        if unknown:
            raise ValueError(f"Unknown model(s) in --skip_models: {unknown}. Available: {all_names}")
        zoo = [z for z in zoo if z[0] not in set(args.skip_models)]

    if not zoo:
        raise ValueError("After filtering, no models remain to run.")

    # CV/search
    cv = KFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_SEED)
    scoring = "neg_root_mean_squared_error"

    results = []
    best_models = {}

    for name, pipeline, param_space in zoo:
        print(f"\n===== Searching: {name} =====")

        if not param_space:
            pipeline.fit(X_trval, y_trval)
            yhat_test = pipeline.predict(X_test)
            test_metrics = evaluate(y_test, yhat_test)
            results.append({
                "model": name,
                "search": "none",
                **{f"test_{k}": v for k, v in test_metrics.items()},
                "best_params": "{}",
            })
            best_models[name] = pipeline
            joblib.dump(pipeline, os.path.join(args.out_dir, f"{name}_best.joblib"))
            print("Test:", test_metrics)
            continue

        n_iter = args.n_iter
        if name in ("svr_rbf", "krr_rbf"):
            n_iter = min(n_iter, 35)  # keep bounded unless you really want pain

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_space,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=RANDOM_SEED,
            n_jobs=args.n_jobs,
            verbose=1,
        )

        search.fit(X_trval, y_trval)
        best = search.best_estimator_
        best_models[name] = best

        yhat_test = best.predict(X_test)
        test_metrics = evaluate(y_test, yhat_test)

        results.append({
            "model": name,
            "search": f"random({n_iter})_cv{args.cv}",
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "best_params": json.dumps(search.best_params_),
        })

        print(f"Best CV score (neg RMSE): {search.best_score_:.4f}")
        print("Test:", test_metrics)

        joblib.dump(best, os.path.join(args.out_dir, f"{name}_best.joblib"))

    res_df = pd.DataFrame(results).sort_values(by="test_rmse", ascending=True)
    res_df.to_csv(os.path.join(args.out_dir, "model_zoo_results_ranked.csv"), index=False)

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "data_csv": args.data_csv,
            "n_rows": int(len(df)),
            "n_features": int(len(numeric_cols)),
            "geo_mode": args.geo_mode,
            "add_canopy_feats": bool(args.add_canopy_feats),
            "cv": int(args.cv),
            "n_iter": int(args.n_iter),
            "n_jobs": int(args.n_jobs),
            "models_ran": [r["model"] for r in results],
            "results": results,
        }, f, indent=2)

    print("\n===== DONE =====")
    print("Saved ranked results to:", os.path.join(args.out_dir, "model_zoo_results_ranked.csv"))
    print("Top 10 by test RMSE:")
    print(res_df.head(10)[["model", "test_rmse", "test_mae", "test_r2", "search"]])


if __name__ == "__main__":
    main()
