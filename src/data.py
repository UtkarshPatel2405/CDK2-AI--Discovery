from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import gdown
import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.config import (
    CURATED_DATA_PATH,
    LEGACY_DATA_PATH,
    LEGACY_MODEL_PATH,
    MODEL_DRIVE_FILE_ID,
    REQUIRED_DATASET_COLUMNS,
)

logger = logging.getLogger(__name__)

def download_model_from_drive(
    file_id: str = MODEL_DRIVE_FILE_ID,
    dest: Optional[Path] = None,
) -> Path:
    
    if dest is None:
        dest = LEGACY_MODEL_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading model from Google Drive (id=%s)  %s", file_id, dest)
    gdown.download(id=file_id, output=str(dest), quiet=False)
    return dest

def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    
    candidates = [p for p in [path, CURATED_DATA_PATH, LEGACY_DATA_PATH] if p is not None]

    data_path = None
    for candidate in candidates:
        if candidate.exists():
            data_path = candidate
            break

    if data_path is None:
        raise FileNotFoundError(
            f"Dataset not found. Searched: {[str(c) for c in candidates]}"
        )

    logger.info("Loading dataset from %s", data_path)
    df = pd.read_parquet(data_path)

    missing = REQUIRED_DATASET_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    df = df.copy()
    df["pic50"] = pd.to_numeric(df["pic50"], errors="coerce")
    df["n_measurements"] = (
        pd.to_numeric(df["n_measurements"], errors="coerce").fillna(0).astype(int)
    )
    if "ic50_nM" in df.columns:
        df["ic50_nM"] = pd.to_numeric(df["ic50_nM"], errors="coerce")

    logger.info(
        "Loaded %d compounds (pIC50 range: %.2f–%.2f)",
        len(df),
        df["pic50"].min(),
        df["pic50"].max(),
    )
    return df

def create_pains_catalog() -> FilterCatalog:
    
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)

def add_scaffold_column(df: pd.DataFrame) -> pd.DataFrame:
   
    out = df.copy()
    scaffolds = []
    for smi in out["smiles"].astype(str).tolist():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffolds.append("")
            continue
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffolds.append(Chem.MolToSmiles(scaffold) if scaffold is not None else "")
    out["murcko_scaffold"] = scaffolds
    return out

def scaffold_stats(
    df_scaf: pd.DataFrame,
    scaffold_smiles: str,
) -> tuple[int, float, float, float, pd.DataFrame]:
    
    sub = (
        df_scaf[df_scaf["murcko_scaffold"] == scaffold_smiles]
        .dropna(subset=["pic50"])
        .copy()
    )
    if len(sub) == 0:
        return 0, np.nan, np.nan, np.nan, pd.DataFrame()

    count = len(sub)
    p_min = float(sub["pic50"].min())
    p_med = float(sub["pic50"].median())
    p_max = float(sub["pic50"].max())

    top = sub.sort_values("pic50", ascending=False).head(5)
    top_view = top[["molecule_chembl_id", "pic50", "n_measurements", "smiles"]].copy()
    top_view["chembl_url"] = (
        top_view["molecule_chembl_id"]
        .astype(str)
        .apply(lambda x: _chembl_molecule_url(x) if x else "")
    )
    return count, p_min, p_med, p_max, top_view

def chembl_pref_name(chembl_id: str) -> Optional[str]:
    
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        name = r.json().get("pref_name")
        return str(name) if name else None
    except Exception:
        logger.debug("ChEMBL API lookup failed for %s", chembl_id)
        return None

def _chembl_molecule_url(chembl_id: str) -> str:
   
    return f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/"
