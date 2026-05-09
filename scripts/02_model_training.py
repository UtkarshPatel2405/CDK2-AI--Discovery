
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.svm import SVR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CV_FOLDS,
    CURATED_DATA_PATH,
    FINAL_MODEL_PATH,
    LEGACY_DATA_PATH,
    RANDOM_STATE,
    SPLITS_DIR,
    TRAINED_MODELS_DIR,
)
from src.model import (
    cross_validate,
    evaluate_model,
    hyperparameter_search,
    save_model,
    smiles_to_feature_matrix,
    train_random_forest,
    y_randomisation_test,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Scaffold-based train/test split
# ------------------------------------------------------------------ #
def scaffold_split(smiles_list: list[str], test_size: float = 0.20, seed: int = RANDOM_STATE):

    from collections import defaultdict

    n = len(smiles_list)
    logger.info("Computing Murcko scaffolds for %d molecules...", n)

    # Step 1: Batch-parse all SMILES into molecules
    mols = [Chem.MolFromSmiles(str(smi)) for smi in smiles_list]

    # Step 2: Vectorised scaffold computation from pre-parsed molecules
    scaffolds: dict[str, list[int]] = defaultdict(list)
    for idx in range(n):
        mol = mols[idx]
        if mol is None:
            scaffolds["INVALID"].append(idx)
        else:
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            scaf = Chem.MolToSmiles(sc) if sc else "NONE"
            scaffolds[scaf].append(idx)
        if (idx + 1) % 500 == 0:
            logger.info("  Scaffolded %d / %d molecules...", idx + 1, n)

    logger.info("  Found %d unique scaffolds", len(scaffolds))

    # Sort scaffolds by size (largest first) for deterministic split
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)

    train_idx, test_idx = [], []
    train_target = int(n * (1 - test_size))

    for s_idx in scaffold_sets:
        if len(train_idx) < train_target:
            train_idx.extend(s_idx)
        else:
            test_idx.extend(s_idx)

    rng = np.random.RandomState(seed)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    logger.info(
        "Scaffold split: %d train, %d test (%.1f%% test)",
        len(train_idx), len(test_idx), 100 * len(test_idx) / n,
    )
    return np.array(train_idx), np.array(test_idx)


def benchmark_models(X_train, y_train, X_test, y_test, cv=CV_FOLDS):

    from scipy import stats as sp_stats

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_features="sqrt", min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.1),
    }

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=1, verbosity=0,
        )
    except ImportError:
        logger.warning("XGBoost not installed — skipping from benchmark")

    results = []
    for name, model in models.items():
        logger.info("--- Benchmarking: %s ---", name)
        t0 = time.time()

        
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = mean_absolute_error(y_test, y_pred)
        pr, _ = sp_stats.pearsonr(y_test, y_pred)
        sr, _ = sp_stats.spearmanr(y_test, y_pred)

        
        from sklearn.model_selection import cross_val_score
        from sklearn.base import clone
        cv_scores = cross_val_score(clone(model), X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)

        result = {
            "model": name,
            "test_r2": round(r2, 4),
            "test_rmse": round(rmse, 4),
            "test_mae": round(mae, 4),
            "test_pearson": round(float(pr), 4),
            "test_spearman": round(float(sr), 4),
            "cv_r2_mean": round(float(cv_scores.mean()), 4),
            "cv_r2_std": round(float(cv_scores.std()), 4),
            "train_time_s": round(train_time, 2),
        }
        results.append(result)
        logger.info("  %s", result)

    return pd.DataFrame(results), models

def main():
    logger.info("=" * 60)
    logger.info("CDK2 MODEL TRAINING PIPELINE")
    logger.info("=" * 60)


    data_path = CURATED_DATA_PATH if CURATED_DATA_PATH.exists() else LEGACY_DATA_PATH
    logger.info("Loading data from %s", data_path)
    df = pd.read_parquet(data_path)
    df["pic50"] = pd.to_numeric(df["pic50"], errors="coerce")
    df = df.dropna(subset=["pic50", "smiles"])
    logger.info("Dataset: %d compounds", len(df))

   
    logger.info("Computing Morgan fingerprints...")
    X, valid_idx = smiles_to_feature_matrix(df["smiles"].tolist())
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    y = df_valid["pic50"].values
    logger.info("Feature matrix: %s, targets: %s", X.shape, y.shape)

   
    train_idx, test_idx = scaffold_split(df_valid["smiles"].tolist())
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

   
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SPLITS_DIR / "train_indices.npy", train_idx)
    np.save(SPLITS_DIR / "test_indices.npy", test_idx)
    logger.info("Saved train/test splits to %s", SPLITS_DIR)

   
    logger.info("\n=== MODEL BENCHMARKING ===")
    benchmark_df, trained_models = benchmark_models(X_train, y_train, X_test, y_test)
    print("\n" + benchmark_df.to_string(index=False))
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(TRAINED_MODELS_DIR / "benchmark_results.csv", index=False)


    logger.info("\n=== RF HYPERPARAMETER SEARCH ===")
    best_rf, best_params = hyperparameter_search(X_train, y_train, n_iter=20)
    logger.info("Best RF params: %s", best_params)

  
    logger.info("\n=== FINAL RF EVALUATION ===")
    final_metrics = evaluate_model(best_rf, X_test, y_test)
    print("\nFinal RF Test Metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v}")

  
    logger.info("\n=== 5-FOLD CROSS-VALIDATION ===")
    cv_results = cross_validate(best_rf, X, y)
    print(f"\nCV R²: {cv_results['mean_r2']:.3f} ± {cv_results['std_r2']:.3f}")
    print(f"CV RMSE: {cv_results['mean_rmse']:.3f} ± {cv_results['std_rmse']:.3f}")
    print(f"CV MAE: {cv_results['mean_mae']:.3f} ± {cv_results['std_mae']:.3f}")

   
    logger.info("\n=== Y-RANDOMISATION TEST ===")
    yrand = y_randomisation_test(best_rf, X, y, n_iterations=10)
    print(f"\nReal CV R²: {yrand['real_cv_r2']:.3f}")
    print(f"Shuffled R²: {yrand['shuffled_r2_mean']:.3f} ± {yrand['shuffled_r2_std']:.3f}")
    print(f"Significant: {yrand['significant']}")

    
    save_model(best_rf, FINAL_MODEL_PATH)

  
    all_results = {
        "dataset_size": len(df_valid),
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "best_rf_params": {k: str(v) for k, v in best_params.items()},
        "test_metrics": final_metrics,
        "cv_r2": f"{cv_results['mean_r2']:.3f} ± {cv_results['std_r2']:.3f}",
        "cv_rmse": f"{cv_results['mean_rmse']:.3f} ± {cv_results['std_rmse']:.3f}",
        "y_randomisation_significant": yrand["significant"],
    }
    with open(TRAINED_MODELS_DIR / "training_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("\n=== TRAINING COMPLETE ===")
    logger.info("Model saved to: %s", FINAL_MODEL_PATH)
    logger.info("Results saved to: %s", TRAINED_MODELS_DIR)


if __name__ == "__main__":
    main()
