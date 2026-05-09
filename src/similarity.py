
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

from src.config import SIM_HIGH, SIM_MED
from src.descriptors import morgan_fp

logger = logging.getLogger(__name__)

def ad_band(max_sim: float) -> str:

    if max_sim >= SIM_HIGH:
        return "In-domain"
    if max_sim >= SIM_MED:
        return "Borderline"
    return "Out-of-domain risk"

def build_dataset_fps(
    smiles_series: pd.Series,
) -> tuple[list[DataStructs.ExplicitBitVect], np.ndarray]:
    
    smiles_list = smiles_series.astype(str).tolist()

    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]

    fps: list[DataStructs.ExplicitBitVect] = []
    idx: list[int] = []

    for i, mol in enumerate(mols):
        if mol is not None:
            fps.append(morgan_fp(mol))
            idx.append(i)

    logger.info("Built %d fingerprints from %d SMILES", len(fps), len(smiles_list))
    return fps, np.array(idx, dtype=int)

def compute_similarity_and_neighbors(
    query_fp: DataStructs.ExplicitBitVect,
    dataset_fps: list[DataStructs.ExplicitBitVect],
    idx_map: np.ndarray,
    df_evidence: pd.DataFrame,
    topk: int = 10,
) -> tuple[float, float, int, int, int, pd.DataFrame]:
   
    if len(dataset_fps) == 0:
        return 0.0, 0.0, 0, 0, 0, pd.DataFrame()

    sims = np.array(
        DataStructs.BulkTanimotoSimilarity(query_fp, dataset_fps),
        dtype=float,
    )

    max_sim = float(sims.max()) if len(sims) else 0.0
    mean_sim = float(sims.mean()) if len(sims) else 0.0
    n_ge_05 = int((sims >= 0.5).sum())
    n_ge_04 = int((sims >= 0.4).sum())
    n_ge_03 = int((sims >= 0.3).sum())

    top_indices = np.argsort(-sims)[:topk]
    rows = []
    for j in top_indices:
        row = df_evidence.iloc[int(idx_map[j])]
        chembl_id = str(row.get("molecule_chembl_id", "")).strip()
        rows.append(
            {
                "similarity": float(sims[j]),
                "molecule_chembl_id": chembl_id,
                "chembl_url": _chembl_molecule_url(chembl_id) if chembl_id else "",
                "pic50_exp": float(row.get("pic50", np.nan)),
                "ic50_nM_exp": float(row.get("ic50_nM", np.nan)),
                "n_measurements": int(row.get("n_measurements", 0)),
                "smiles": str(row.get("smiles", "")),
            }
        )

    return max_sim, mean_sim, n_ge_05, n_ge_04, n_ge_03, pd.DataFrame(rows)

def _chembl_molecule_url(chembl_id: str) -> str:

    return f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/"
