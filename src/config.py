from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
FIGURES_DIR = MODELS_DIR / "figures"

LEGACY_DATA_PATH = PROJECT_ROOT / "cdk2_pic50_clean.parquet"
LEGACY_MODEL_PATH = PROJECT_ROOT / "cdk2_rf_final_all_data.joblib"

CURATED_DATA_PATH = PROCESSED_DATA_DIR / "cdk2_pic50_curated.parquet"
FINAL_MODEL_PATH = TRAINED_MODELS_DIR / "cdk2_rf_final.joblib"

MODEL_DRIVE_FILE_ID = "1pOgZVHG7BfrcXE7ZmHJNM9CMnsBNlCQa"

FP_RADIUS: int = 2
FP_NBITS: int = 2048

SIM_HIGH: float = 0.50
SIM_MED: float = 0.30
SIM_NEIGHBOR: float = 0.40

DEFAULT_TRIAGE_PIC50: float = 7.0
DEFAULT_TRIAGE_STD: float = 0.60
DEFAULT_TRIAGE_SIM: float = 0.30

CHEMBL_TARGET_ID = "CHEMBL301"
CHEMBL_TARGET_NAME = "Cyclin-Dependent Kinase 2"

REQUIRED_DATASET_COLUMNS = frozenset({
    "smiles",
    "pic50",
    "n_measurements",
    "inchikey",
    "molecule_chembl_id",
})

RANDOM_STATE: int = 42
CV_FOLDS: int = 5
TEST_SIZE: float = 0.20

RF_DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.3],
}

POTENCY_STRONG_THRESHOLD: float = 8.0
POTENCY_MODERATE_THRESHOLD: float = 6.0

UNCERTAINTY_LOW_THRESHOLD: float = 0.35
UNCERTAINTY_MED_THRESHOLD: float = 0.60

RO5_MW_MAX: float = 500.0
RO5_LOGP_MAX: float = 5.0
RO5_HBD_MAX: int = 5
RO5_HBA_MAX: int = 10

VEBER_TPSA_MAX: float = 140.0
VEBER_ROTB_MAX: int = 10
