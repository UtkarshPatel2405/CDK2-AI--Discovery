
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CHEMBL_TARGET_ID,
    CURATED_DATA_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def fetch_chembl_data(target_id: str = CHEMBL_TARGET_ID) -> pd.DataFrame:

    logger.info("Fetching bioactivity data for target %s from ChEMBL...", target_id)

    try:
        from chembl_webresource_client.new_client import new_client

        activity = new_client.activity
        res = activity.filter(
            target_chembl_id=target_id,
            standard_type="IC50",
            standard_relation="=",
            standard_units="nM",
            assay_type="B",
        ).only(
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_value",
            "standard_type",
            "standard_units",
            "standard_relation",
            "pchembl_value",
            "assay_chembl_id",
            "document_chembl_id",
        )
        df = pd.DataFrame(list(res))
        logger.info("Fetched %d records via chembl_webresource_client", len(df))
        return df

    except ImportError:
        logger.warning("chembl_webresource_client not installed. Using REST API...")
        import requests

        url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
        params = {
            "target_chembl_id": target_id,
            "standard_type": "IC50",
            "standard_relation": "=",
            "standard_units": "nM",
            "assay_type": "B",
            "limit": 1000,
            "offset": 0,
        }
        all_records = []
        while True:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            records = data.get("activities", [])
            if not records:
                break
            all_records.extend(records)
            logger.info("  Fetched %d records so far...", len(all_records))
            if data["page_meta"]["next"] is None:
                break
            params["offset"] += params["limit"]

        df = pd.DataFrame(all_records)
        logger.info("Fetched %d records via REST API", len(df))
        return df

def curate_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:

    logger.info("=== Starting data curation ===")
    log = []

    df = df_raw.copy()
    if "canonical_smiles" in df.columns:
        df = df.rename(columns={"canonical_smiles": "smiles_raw"})

    initial = len(df)
    log.append(f"Initial records: {initial}")

    df = df.dropna(subset=["smiles_raw"])
    df = df[df["smiles_raw"].str.strip() != ""]
    log.append(f"After removing missing SMILES: {len(df)} (removed {initial - len(df)})")

    df["standard_value"] = pd.to_numeric(df.get("standard_value"), errors="coerce")
    before = len(df)
    df = df.dropna(subset=["standard_value"])
    df = df[df["standard_value"] > 0]
    log.append(f"After removing invalid IC50 values: {len(df)} (removed {before - len(df)})")

    raw_smiles_list = df["smiles_raw"].astype(str).tolist()
    n_total = len(raw_smiles_list)
    logger.info("  Batch-parsing %d SMILES...", n_total)

    parsed_mols = [Chem.MolFromSmiles(smi) for smi in raw_smiles_list]

    valid_smiles = []
    valid_inchikeys = []
    invalid_count = 0

    for i, mol in enumerate(parsed_mols):
        if mol is None:
            invalid_count += 1
            valid_smiles.append(None)
            valid_inchikeys.append(None)
            continue
        try:
            Chem.SanitizeMol(mol)

            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if frags:
                mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
            valid_smiles.append(Chem.MolToSmiles(mol, canonical=True))
            valid_inchikeys.append(Chem.inchi.MolToInchiKey(mol))
        except Exception:
            invalid_count += 1
            valid_smiles.append(None)
            valid_inchikeys.append(None)

        if (i + 1) % 500 == 0:
            logger.info("  Processed %d / %d SMILES...", i + 1, n_total)

    df["smiles"] = valid_smiles
    df["inchikey"] = valid_inchikeys
    df = df.dropna(subset=["smiles", "inchikey"])
    log.append(f"After RDKit parsing + salt stripping: {len(df)} (invalid: {invalid_count})")

    df["ic50_nM"] = df["standard_value"].astype(float)
    df["pic50"] = -np.log10(df["ic50_nM"] * 1e-9)
    log.append(f"Converted IC50 (nM)  pIC50")

    before = len(df)
    df = df[(df["pic50"] >= 3.0) & (df["pic50"] <= 12.0)]
    log.append(f"After outlier removal (3  pIC50  12): {len(df)} (removed {before - len(df)})")

    before = len(df)
    agg = (
        df.groupby("inchikey")
        .agg(
            smiles=("smiles", "first"),
            pic50=("pic50", "median"),
            ic50_nM=("ic50_nM", "median"),
            n_measurements=("pic50", "count"),
            molecule_chembl_id=("molecule_chembl_id", "first"),
        )
        .reset_index()
    )
    log.append(f"After aggregation by InChIKey (median): {len(agg)} unique compounds (from {before} records)")

    smiles_for_scaffolds = agg["smiles"].tolist()
    logger.info("  Computing Murcko scaffolds for %d compounds...", len(smiles_for_scaffolds))

    scaffold_mols = [Chem.MolFromSmiles(smi) for smi in smiles_for_scaffolds]
    scaffolds = []
    for mol in scaffold_mols:
        if mol is None:
            scaffolds.append("")
        else:
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.append(Chem.MolToSmiles(sc) if sc else "")
    agg["murcko_scaffold"] = scaffolds

    n_scaffolds = agg["murcko_scaffold"].nunique()
    log.append(f"Unique Murcko scaffolds: {n_scaffolds}")

    logger.info("Curation Summary")
    for line in log:
        logger.info("  %s", line)

    return agg

def save_dataset(df: pd.DataFrame, path: Path = CURATED_DATA_PATH) -> None:
   
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved curated dataset to %s (%d compounds)", path, len(df))

def print_summary_statistics(df: pd.DataFrame) -> None:
    
    print("\n" + "=" * 60)
    print("DATASET SUMMARY (for paper Methods section)")
    print("=" * 60)
    print(f"  Total compounds:        {len(df)}")
    print(f"  pIC50 range:            {df['pic50'].min():.2f} – {df['pic50'].max():.2f}")
    print(f"  pIC50 mean ± std:       {df['pic50'].mean():.2f} ± {df['pic50'].std():.2f}")
    print(f"  pIC50 median:           {df['pic50'].median():.2f}")
    print(f"  Unique scaffolds:       {df['murcko_scaffold'].nunique()}")
    print(f"  Multi-measurement:      {(df['n_measurements'] > 1).sum()} compounds")
    print(f"  Max measurements:       {df['n_measurements'].max()}")
    print()
    print("  pIC50 distribution:")
    for low, high, label in [(3, 5, "Weak (<5)"), (5, 6, "Low (5-6)"),
                              (6, 7, "Moderate (6-7)"), (7, 8, "Good (7-8)"),
                              (8, 12, "Strong (8)")]:
        n = len(df[(df["pic50"] >= low) & (df["pic50"] < high)])
        print(f"    {label}: {n} ({100*n/len(df):.1f}%)")
    print("=" * 60)

def main():

    raw_cache = RAW_DATA_DIR / "cdk2_raw_chembl.csv"
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if raw_cache.exists():
        logger.info("Loading cached raw data from %s", raw_cache)
        df_raw = pd.read_csv(raw_cache)
    else:
        df_raw = fetch_chembl_data()
        df_raw.to_csv(raw_cache, index=False)
        logger.info("Cached raw data to %s", raw_cache)

    df_curated = curate_dataset(df_raw)

    save_dataset(df_curated)

    print_summary_statistics(df_curated)

    logger.info("Data curation complete!")

if __name__ == "__main__":
    main()
