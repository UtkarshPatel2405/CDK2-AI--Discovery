
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score

from src.config import CV_FOLDS, FP_NBITS, FP_RADIUS, LEGACY_MODEL_PATH, RANDOM_STATE, RF_DEFAULT_PARAMS, RF_PARAM_GRID
from src.descriptors import fp_to_array, morgan_fp

logger = logging.getLogger(__name__)

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)

def load_model(path: Optional[Path] = None) -> Any:

    if path is None:
        path = LEGACY_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)

def rf_predict(model: RandomForestRegressor, mol: Chem.Mol) -> tuple[float, float, Any]:
    
    fp = morgan_fp(mol)
    X = fp_to_array(fp).reshape(1, -1)
    tree_preds = np.array([t.predict(X)[0] for t in model.estimators_], dtype=float)
    return float(tree_preds.mean()), float(tree_preds.std(ddof=1)), fp

def batch_parse_smiles(smiles_list: list[str]) -> tuple[list[Chem.Mol], list[int]]:
    
    n = len(smiles_list)
    valid_mols: list[Chem.Mol] = []
    valid_idx: list[int] = []

    for i in range(n):
        mol = Chem.MolFromSmiles(str(smiles_list[i]))
        if mol is not None:
            valid_mols.append(mol)
            valid_idx.append(i)
        if (i + 1) % 500 == 0:
            logger.info("  Parsed %d / %d SMILES (%d valid)...", i + 1, n, len(valid_mols))

    logger.info("  Parsing complete: %d / %d valid molecules", len(valid_mols), n)
    return valid_mols, valid_idx

def smiles_to_feature_matrix(smiles_list: list[str]) -> tuple[np.ndarray, list[int]]:
    
    valid_mols, valid_idx = batch_parse_smiles(smiles_list)

    if not valid_mols:
        return np.empty((0, FP_NBITS), dtype=np.uint8), []

    n = len(valid_mols)

    X = np.zeros((n, FP_NBITS), dtype=np.uint8)

    logger.info("  Generating %d fingerprints into pre-allocated matrix...", n)
    for i, mol in enumerate(valid_mols):
        fp = _FP_GEN.GetFingerprintAsNumPy(mol)
        X[i] = fp
        if (i + 1) % 500 == 0:
            logger.info("  Fingerprinted %d / %d molecules...", i + 1, n)

    logger.info("  Feature matrix ready: shape=%s, dtype=%s", X.shape, X.dtype)
    return X, valid_idx

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, params: Optional[dict] = None) -> RandomForestRegressor:
    
    if params is None:
        params = RF_DEFAULT_PARAMS.copy()
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model

def hyperparameter_search(X_train, y_train, n_iter=20, cv=CV_FOLDS, scoring="neg_root_mean_squared_error"):
    
    logger.info("HP search: %d iterations, %d-fold CV", n_iter, cv)

    base = RandomForestRegressor(random_state=RANDOM_STATE)
    search = RandomizedSearchCV(base, RF_PARAM_GRID, n_iter=n_iter, cv=cv, scoring=scoring,
                                random_state=RANDOM_STATE, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    logger.info("Best score: %.4f | Best params: %s", search.best_score_, search.best_params_)
    return search.best_estimator_, search.best_params_

def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    
    from scipy import stats
    y_pred = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "pearson_r": float(stats.pearsonr(y_test, y_pred)[0]),
        "spearman_rho": float(stats.spearmanr(y_test, y_pred)[0]),
        "n_test": len(y_test),
    }

def cross_validate(model, X, y, cv=CV_FOLDS) -> dict[str, Any]:
   
    from sklearn.base import clone
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics, y_pred_all = [], np.zeros_like(y, dtype=float)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        fm = clone(model)
        fm.fit(X[train_idx], y[train_idx])
        yp = fm.predict(X[val_idx])
        y_pred_all[val_idx] = yp
        fold_metrics.append({"fold": fold_idx+1, "r2": float(r2_score(y[val_idx], yp)),
                             "rmse": float(np.sqrt(mean_squared_error(y[val_idx], yp))),
                             "mae": float(mean_absolute_error(y[val_idx], yp))})
    r2s = [m["r2"] for m in fold_metrics]
    rmses = [m["rmse"] for m in fold_metrics]
    maes = [m["mae"] for m in fold_metrics]
    return {"per_fold": fold_metrics, "mean_r2": float(np.mean(r2s)), "std_r2": float(np.std(r2s)),
            "mean_rmse": float(np.mean(rmses)), "std_rmse": float(np.std(rmses)),
            "mean_mae": float(np.mean(maes)), "std_mae": float(np.std(maes)),
            "cv_predictions": y_pred_all}

def y_randomisation_test(model, X, y, n_iterations=10, cv=CV_FOLDS) -> dict[str, Any]:
    
    from sklearn.base import clone
    rng = np.random.RandomState(RANDOM_STATE)
    shuffled_r2 = []
    for it in range(n_iterations):
        y_shuf = rng.permutation(y)
        scores = cross_val_score(clone(model), X, y_shuf, cv=cv, scoring="r2", n_jobs=-1)
        shuffled_r2.append(float(np.mean(scores)))
        logger.info("  Y-rand iteration %d/%d: shuffled R²=%.3f", it + 1, n_iterations, shuffled_r2[-1])
    real_scores = cross_val_score(clone(model), X, y, cv=cv, scoring="r2", n_jobs=-1)
    real_r2 = float(np.mean(real_scores))
    sig = real_r2 > float(np.mean(shuffled_r2) + 3 * np.std(shuffled_r2))
    return {"real_cv_r2": real_r2, "shuffled_r2_mean": float(np.mean(shuffled_r2)),
            "shuffled_r2_std": float(np.std(shuffled_r2)), "shuffled_r2_values": shuffled_r2,
            "significant": sig}

def save_model(model, path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)
