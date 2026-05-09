
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CURATED_DATA_PATH,
    FIGURES_DIR,
    FINAL_MODEL_PATH,
    LEGACY_DATA_PATH,
    LEGACY_MODEL_PATH,
    SPLITS_DIR,
    TRAINED_MODELS_DIR,
)
from src.model import cross_validate, evaluate_model, load_model, smiles_to_feature_matrix, y_randomisation_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def plot_pred_vs_exp(y_true, y_pred, save_path: Path):
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pr, _ = sp_stats.pearsonr(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.4, s=25, c="#1f77b4", edgecolors="none")

    lims = [min(y_true.min(), y_pred.min()) - 0.5, max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, "--", color="gray", linewidth=1, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Experimental pIC50")
    ax.set_ylabel("Predicted pIC50")
    ax.set_title("Predicted vs Experimental pIC50")
    ax.set_aspect("equal")

    textstr = f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nr = {pr:.3f}\nn = {len(y_true)}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend(loc="lower right")

    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)

def plot_residuals(y_true, y_pred, save_path: Path):
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.4, s=20, c="#2ca02c", edgecolors="none")
    axes[0].axhline(y=0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted pIC50")
    axes[0].set_ylabel("Residual (Pred - Exp)")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=40, color="#ff7f0e", alpha=0.7, edgecolor="black", linewidth=0.5)
    axes[1].axvline(x=0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual (Pred - Exp)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Residual Distribution (mean={residuals.mean():.3f}, std={residuals.std():.3f})")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)

def plot_activity_distribution(y, save_path: Path):
   
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y, bins=50, color="#9467bd", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(x=np.median(y), color="red", linestyle="--", label=f"Median: {np.median(y):.2f}")
    ax.axvline(x=np.mean(y), color="blue", linestyle="--", label=f"Mean: {np.mean(y):.2f}")
    ax.set_xlabel("pIC50")
    ax.set_ylabel("Count")
    ax.set_title(f"CDK2 pIC50 Distribution (n={len(y)})")
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)

def plot_feature_importance(model, save_path: Path, top_n: int = 30):

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-top_n:][::-1]
    top_imp = importances[top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), top_imp[::-1], color="#17becf", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"Bit {i}" for i in top_idx[::-1]])
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Top {top_n} Morgan Fingerprint Bits by Importance")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)

def plot_cv_folds(cv_results: dict, save_path: Path):

    folds = cv_results["per_fold"]
    fold_nums = [f["fold"] for f in folds]
    r2_vals = [f["r2"] for f in folds]
    rmse_vals = [f["rmse"] for f in folds]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(fold_nums, r2_vals, color="#1f77b4", edgecolor="black", linewidth=0.5)
    axes[0].axhline(y=cv_results["mean_r2"], color="red", linestyle="--",
                    label=f"Mean: {cv_results['mean_r2']:.3f}")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("R²")
    axes[0].set_title("Cross-Validation R² per Fold")
    axes[0].legend()

    axes[1].bar(fold_nums, rmse_vals, color="#ff7f0e", edgecolor="black", linewidth=0.5)
    axes[1].axhline(y=cv_results["mean_rmse"], color="red", linestyle="--",
                    label=f"Mean: {cv_results['mean_rmse']:.3f}")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Cross-Validation RMSE per Fold")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)

def main():
    logger.info("=" * 60)
    logger.info("CDK2 MODEL EVALUATION & FIGURE GENERATION")
    logger.info("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data_path = CURATED_DATA_PATH if CURATED_DATA_PATH.exists() else LEGACY_DATA_PATH
    df = pd.read_parquet(data_path)
    df["pic50"] = pd.to_numeric(df["pic50"], errors="coerce")
    df = df.dropna(subset=["pic50", "smiles"])

    X, valid_idx = smiles_to_feature_matrix(df["smiles"].tolist())
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    y = df_valid["pic50"].values

    model_path = FINAL_MODEL_PATH if FINAL_MODEL_PATH.exists() else LEGACY_MODEL_PATH
    model = load_model(model_path)
    logger.info("Loaded model from %s", model_path)

    train_path = SPLITS_DIR / "train_indices.npy"
    test_path = SPLITS_DIR / "test_indices.npy"
    if train_path.exists() and test_path.exists():
        train_idx = np.load(train_path)
        test_idx = np.load(test_path)

        train_idx = train_idx[train_idx < len(X)]
        test_idx = test_idx[test_idx < len(X)]
    else:
        logger.warning("No saved splits found, using random 80/20 split")
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(X))
        split = int(0.8 * len(perm))
        train_idx, test_idx = perm[:split], perm[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    y_pred_test = model.predict(X_test)

    plot_pred_vs_exp(y_test, y_pred_test, FIGURES_DIR / "fig1_pred_vs_exp.png")

    plot_residuals(y_test, y_pred_test, FIGURES_DIR / "fig2_residuals.png")

    plot_activity_distribution(y, FIGURES_DIR / "fig3_activity_distribution.png")

    if hasattr(model, "feature_importances_"):
        plot_feature_importance(model, FIGURES_DIR / "fig5_feature_importance.png")

    logger.info("Running cross-validation...")
    cv_results = cross_validate(model, X, y)
    plot_cv_folds(cv_results, FIGURES_DIR / "fig7_cv_folds.png")

    plot_pred_vs_exp(y, cv_results["cv_predictions"], FIGURES_DIR / "fig1b_cv_pred_vs_exp.png")

    test_metrics = evaluate_model(model, X_test, y_test)

    report = {
        "dataset": str(data_path.name),
        "n_total": len(df_valid),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "test_metrics": test_metrics,
        "cv_r2": f"{cv_results['mean_r2']:.3f} ± {cv_results['std_r2']:.3f}",
        "cv_rmse": f"{cv_results['mean_rmse']:.3f} ± {cv_results['std_rmse']:.3f}",
        "cv_mae": f"{cv_results['mean_mae']:.3f} ± {cv_results['std_mae']:.3f}",
    }

    report_path = TRAINED_MODELS_DIR / "evaluation_report.json"
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))
    print("=" * 60)
    print(f"\nFigures saved to: {FIGURES_DIR}")

    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()
