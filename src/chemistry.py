from __future__ import annotations

import logging
from io import BytesIO
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Draw

logger = logging.getLogger(__name__)

def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:

    if not smiles or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        logger.debug("Failed to parse SMILES: %s", smiles)
        return None

def canonical_smiles(mol: Chem.Mol) -> str:
    
    return Chem.MolToSmiles(mol, canonical=True)

def smiles_to_canonical(smiles: str) -> Optional[str]:
   
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return canonical_smiles(mol)

def inchikey(mol: Chem.Mol) -> str:
   
    try:
        return Chem.inchi.MolToInchiKey(mol)
    except Exception:
        logger.warning("InChIKey generation failed")
        return "N/A"

def keep_largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())

def mol_to_png_bytes(mol: Chem.Mol, size: tuple[int, int] = (520, 340)) -> bytes:
    
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
