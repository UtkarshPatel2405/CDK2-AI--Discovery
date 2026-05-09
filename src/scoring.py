
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Optional
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.chemistry import canonical_smiles, inchikey, keep_largest_fragment, mol_from_smiles
from src.config import SIM_MED
from src.data import _chembl_molecule_url, chembl_pref_name, scaffold_stats
from src.descriptors import (
    calculate_ligand_efficiency, check_pains, compute_physicochemical,
    morgan_fp, pic50_to_ic50_nM, potency_band, ro5_violations,
    uncertainty_band, veber_pass,
)
from src.model import rf_predict
from src.similarity import ad_band, build_dataset_fps, compute_similarity_and_neighbors

@dataclass
class BatchRow:

    ok: bool
    error: str
    id: str
    smiles_in: str
    smiles_used: str
    pred_pic50: float
    pred_ic50_nM: float
    pred_std: float
    potency: str
    ligand_efficiency: float
    max_sim: float
    mean_sim: float
    n_sim_ge_0_5: int
    n_sim_ge_0_4: int
    n_sim_ge_0_3: int
    ad_band: str
    pains_count: int
    pains_hits: str
    scaffold_smiles: str
    scaffold_count_in_dataset: int
    ro5_violations: int
    veber_pass: bool
    molwt: float
    clogp: float
    tpsa: float
    rotb: int
    hbd: int
    hba: int
    rings: int
    qed: float
    chembl_id: str
    chembl_name: str
    chembl_url: str
    priority: str
    pass_triage: bool

def _empty_row(row_id: str, smiles_in: str, error: str) -> BatchRow:

    return BatchRow(
        False, error, row_id, smiles_in, "", np.nan, np.nan, np.nan, "", np.nan,
        np.nan, np.nan, 0, 0, 0, "", 0, "", "", 0, 0, False,
        np.nan, np.nan, np.nan, 0, 0, 0, 0, np.nan, "", "", "", "", False,
    )

def make_priority(pred_pic50: float, pred_std: float, max_sim: float, pains_hits: list[str]) -> str:

    score = 0
    score += 2 if pred_pic50 >= 7.0 else (1 if pred_pic50 >= 6.0 else 0)
    score += 1 if pred_std <= 0.60 else 0
    score += 1 if max_sim >= 0.30 else 0
    score -= 2 if pains_hits else 0
    if score >= 3:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"

def decision_text(pred_pic50: float, pred_std: float, max_sim: float, le: float, pains_hits: list[str]):

    rationale = [
        f"Potency: {potency_band(pred_pic50)} (pIC50={pred_pic50:.2f}, IC50{pic50_to_ic50_nM(pred_pic50):.1f} nM).",
        f"Uncertainty σ: {pred_std:.3f} ({uncertainty_band(pred_std)}).",
        f"Applicability domain: max sim={max_sim:.3f} ({ad_band(max_sim)}).",
        f"Ligand Efficiency: LE={le:.2f} (0.30 acceptable; 0.35 strong).",
    ]
    next_steps = []
    if pains_hits:
        rationale.append(f"PAINS alerts: {', '.join(pains_hits)}.")
        next_steps.append("Run orthogonal counterscreens to rule out assay interference.")
        next_steps.append("Consider redesign to remove flagged PAINS motifs.")
    if max_sim < SIM_MED:
        next_steps.append("Out-of-domain/borderline: validate experimentally before heavy optimization.")
        next_steps.append("Search for closer analogs to improve domain support.")
    if pred_pic50 >= 7.0:
        next_steps.append("Proceed to selectivity profiling (CDK1/4/6) + early ADME (solubility, stability).")
    else:
        next_steps.append("Consider analog design around scaffold to improve potency and reduce risk flags.")
    return rationale, next_steps

def score_smiles_row(
    smiles: str, row_id: str, *, model, pains_catalog,
    df_evidence: pd.DataFrame, df_full: pd.DataFrame, df_scaf: pd.DataFrame,
    dataset_fps=None, dataset_idx_map=None,
    strip_salts: bool, compute_ad_flag: bool, topk_neighbors: int,
    triage_pic50: float, triage_std: float, triage_sim: float,
) -> BatchRow:

    smiles_in = (smiles or "").strip()
    if not smiles_in:
        return _empty_row(row_id, "", "Empty SMILES")

    mol = mol_from_smiles(smiles_in)
    if mol is None:
        return _empty_row(row_id, smiles_in, "Invalid SMILES")

    if strip_salts:
        mol = keep_largest_fragment(mol)

    smiles_used = canonical_smiles(mol)
    pred_pic50, pred_std, fp = rf_predict(model, mol)
    pred_ic50 = pic50_to_ic50_nM(pred_pic50)
    le = calculate_ligand_efficiency(pred_pic50, mol)
    pains = check_pains(mol, pains_catalog)
    phys = compute_physicochemical(mol)

    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    scaf_smi = Chem.MolToSmiles(scaffold_mol) if scaffold_mol else ""
    sc_count = 0
    if scaf_smi:
        sc_count, *_ = scaffold_stats(df_scaf, scaf_smi)

    cid, cname, curl = "", "", ""
    ik = inchikey(mol)
    m = df_full[df_full["inchikey"].astype(str) == ik]
    if len(m) > 0:
        cid = str(m.iloc[0].get("molecule_chembl_id", "")).strip()
        if cid:
            cname = chembl_pref_name(cid) or ""
            curl = _chembl_molecule_url(cid)

    max_sim = mean_sim = 0.0
    n05 = n04 = n03 = 0
    if compute_ad_flag and dataset_fps is not None:
        max_sim, mean_sim, n05, n04, n03, _ = compute_similarity_and_neighbors(
            fp, dataset_fps, dataset_idx_map, df_evidence, topk_neighbors
        )

    priority = make_priority(pred_pic50, pred_std, max_sim if compute_ad_flag else 0.0, pains)
    pass_triage = bool(
        (pred_pic50 >= triage_pic50)
        and (pred_std <= triage_std)
        and ((max_sim >= triage_sim) if compute_ad_flag else True)
        and (len(pains) == 0)
    )

    return BatchRow(
        True, "", str(row_id), smiles_in, smiles_used,
        float(pred_pic50), float(pred_ic50), float(pred_std),
        potency_band(pred_pic50), float(le),
        float(max_sim), float(mean_sim), int(n05), int(n04), int(n03),
        ad_band(max_sim) if compute_ad_flag else "",
        int(len(pains)), "; ".join(pains), scaf_smi, int(sc_count),
        int(ro5_violations(mol)), bool(veber_pass(mol)),
        phys.mol_wt, phys.clogp, phys.tpsa, phys.rotatable_bonds,
        phys.hbd, phys.hba, phys.num_rings, phys.qed,
        cid, cname, curl, priority, pass_triage,
    )
