
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors

from src.config import (
    FP_NBITS,
    FP_RADIUS,
    RO5_HBA_MAX,
    RO5_HBD_MAX,
    RO5_LOGP_MAX,
    RO5_MW_MAX,
    VEBER_ROTB_MAX,
    VEBER_TPSA_MAX,
)

def morgan_fp(
    mol: Chem.Mol,
    radius: int = FP_RADIUS,
    n_bits: int = FP_NBITS,
) -> DataStructs.ExplicitBitVect:
    
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

def fp_to_array(fp: DataStructs.ExplicitBitVect) -> np.ndarray:
    
    n_bits = fp.GetNumBits()
    arr = np.zeros(n_bits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

@dataclass(frozen=True)
class PhysicoChemProfile:
   
    mol_wt: float
    clogp: float
    tpsa: float
    rotatable_bonds: int
    hbd: int
    hba: int
    num_rings: int
    num_heavy_atoms: int
    qed: float

def compute_physicochemical(mol: Chem.Mol) -> PhysicoChemProfile:
    
    return PhysicoChemProfile(
        mol_wt=float(Descriptors.MolWt(mol)),
        clogp=float(Descriptors.MolLogP(mol)),
        tpsa=float(rdMolDescriptors.CalcTPSA(mol)),
        rotatable_bonds=int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        hbd=int(rdMolDescriptors.CalcNumHBD(mol)),
        hba=int(rdMolDescriptors.CalcNumHBA(mol)),
        num_rings=int(rdMolDescriptors.CalcNumRings(mol)),
        num_heavy_atoms=int(mol.GetNumHeavyAtoms()),
        qed=float(QED.qed(mol)),
    )

def ro5_violations(mol: Chem.Mol) -> int:
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)

    violations = 0
    if mw > RO5_MW_MAX:
        violations += 1
    if logp > RO5_LOGP_MAX:
        violations += 1
    if hbd > RO5_HBD_MAX:
        violations += 1
    if hba > RO5_HBA_MAX:
        violations += 1
    return violations

def veber_pass(mol: Chem.Mol) -> bool:
    
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return (tpsa <= VEBER_TPSA_MAX) and (rotb <= VEBER_ROTB_MAX)

def calculate_ligand_efficiency(pic50: float, mol: Chem.Mol) -> float:
    
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy == 0:
        return 0.0
    return 1.37 * pic50 / n_heavy

def check_pains(
    mol: Chem.Mol,
    catalog: "FilterCatalog",
) -> list[str]:
    
    entries = catalog.GetMatches(mol)
    return [e.GetDescription() for e in entries]

def pic50_to_ic50_nM(pic50: float) -> float:
    
    return float(10 ** (9.0 - pic50))

def potency_band(pic50: float) -> str:
    
    if pic50 >= 8.0:
        return "Strong"
    if pic50 >= 6.0:
        return "Moderate"
    return "Weak"

def uncertainty_band(std: float) -> str:
   
    if std <= 0.35:
        return "Low (good)"
    if std <= 0.60:
        return "Medium"
    return "High (warning)"
