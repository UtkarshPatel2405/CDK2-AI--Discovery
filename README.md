# ChemBLitz — CDK2 Pharmacological Diagnostic Suite

[![Live App](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?logo=streamlit)](https://cdk2-ai-drug-discovery-ffr8hm9thsadgagahfhjy6.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

ChemBLitz is a professional decision-support platform for **CDK2 inhibitor discovery**. It integrates a validated **QSAR predictor** with comprehensive **chemical asset validation** — including applicability domain analysis, nearest-neighbour evidence, PAINS screening, scaffold context, and developability profiling — to support compound prioritisation in hit-to-lead and lead optimisation workflows.

---

## Table of Contents

- [What This Tool Does](#what-this-tool-does)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Reproducible Pipeline](#reproducible-pipeline)
- [How to Interpret Results](#how-to-interpret-results)
- [Methodology](#methodology)
- [References](#references)
- [Citation](#citation)
- [Disclaimer](#disclaimer)

---

## What This Tool Does

### 1) QSAR Prediction (CDK2 pIC50)
- Predicts **pIC50** (and IC50 in nM) using a Random Forest model trained on Morgan fingerprints (ECFP4-equivalent)
- Reports **uncertainty (σ)** from tree-to-tree disagreement as a model confidence proxy

### 2) Chemical Asset Validation
- **Applicability Domain (AD):** Tanimoto similarity to training chemistry with multi-threshold assessment
- **Nearest-neighbour evidence:** Top-K similar compounds with experimental pIC50/IC50
- **PAINS screening:** Flags common assay-interference substructures
- **Scaffold context:** Murcko scaffold frequency + activity distribution + top exemplars
- **Developability profiling:** Lipinski Ro5, Veber rules, QED, ligand efficiency, and property risk indicators

### 3) Library Triage (CSV)
- Batch-scores compound libraries with configurable pass/fail criteria
- Exports full results and filtered hit lists for downstream workflows

---

## Repository Structure

```
├── src/                          # Core library (framework-agnostic)
│   ├── config.py                 # Constants, paths, thresholds (with literature citations)
│   ├── chemistry.py              # Molecule parsing, canonicalization, rendering
│   ├── descriptors.py            # Fingerprints, physicochemical properties, drug-likeness
│   ├── similarity.py             # Tanimoto similarity, AD assessment, neighbor search
│   ├── data.py                   # Data loading, scaffold analysis, ChEMBL API
│   ├── model.py                  # Prediction, training, CV, Y-randomisation
│   └── scoring.py                # Batch scoring, triage, priority logic
├── scripts/                      # Reproducible pipeline
│   ├── 01_data_curation.py       # ChEMBL → cleaned dataset
│   ├── 02_model_training.py      # Multi-model benchmarking + scaffold split + CV
│   └── 03_model_evaluation.py    # Evaluation figures + metrics report
├── tests/                        # Unit tests
│   ├── test_chemistry.py
│   ├── test_descriptors.py
│   └── test_scoring.py
├── data/                         # Data directory (raw, processed, splits)
├── models/                       # Trained models + evaluation figures
├── app.py                        # Streamlit dashboard (UI layer only)
├── Makefile                      # Pipeline automation
├── requirements.txt              # Pinned dependencies
└── requirements-dev.txt          # Dev/test dependencies
```

---

## Installation

### Requirements
- Python 3.10+
- RDKit (installed via pip or conda)

### Setup
```bash
# Clone repository
git clone https://github.com/<your-username>/cdk2-ai-drug-discovery.git
cd cdk2-ai-drug-discovery

# Install dependencies
pip install -r requirements.txt

# (Optional) Install dev/test dependencies
pip install -r requirements-dev.txt
```

### Run the App
```bash
streamlit run app.py
```

---

## Reproducible Pipeline

The full modelling pipeline is automated via `make` commands:

```bash
# Step 1: Curate data from ChEMBL
make data

# Step 2: Train & benchmark models (RF, XGBoost, Ridge, SVR)
make train

# Step 3: Generate evaluation figures and metrics
make evaluate

# Run all steps
make pipeline

# Run tests
make test
```

On Windows without `make`, run the scripts directly:
```bash
python scripts/01_data_curation.py
python scripts/02_model_training.py
python scripts/03_model_evaluation.py
```

---

## How to Interpret Results

| Metric | Interpretation |
|--------|---------------|
| **pIC50** | −log₁₀(IC50 in M). +1 pIC50 ≈ 10× stronger potency |
| **IC50 (nM)** | 10^(9 − pIC50). Reported for intuition |
| **σ (uncertainty)** | Std across RF trees. Not a calibrated CI, but a disagreement proxy |
| **AD (max sim)** | ≥0.50: in-domain; 0.30–0.50: borderline; <0.30: out-of-domain risk |
| **LE** | Ligand efficiency. ≥0.30 acceptable; ≥0.35 strong |
| **PAINS** | Assay interference flags. Requires orthogonal validation |

---

## Methodology

### Target
- **CDK2** (Cyclin-Dependent Kinase 2) — ChEMBL: CHEMBL301

### Data Curation
- Source: ChEMBL bioactivity database (IC50, binding assays, exact measurements)
- Cleaning: RDKit sanitisation, salt stripping, canonical SMILES, InChIKey deduplication
- Aggregation: Median pIC50 across duplicate measurements
- Outlier removal: pIC50 ∈ [3, 12]

### Model
- **Features:** Morgan fingerprints (radius 2, 2048 bits) — ECFP4-equivalent
- **Algorithm:** Random Forest with hyperparameter optimisation
- **Split:** Scaffold-based (Murcko) — ensures generalisation to novel chemotypes
- **Validation:** 5-fold cross-validation + Y-randomisation test
- **Benchmarking:** Compared against Ridge, XGBoost, SVR

### Applicability Domain
- Distance-to-model approach using Tanimoto similarity
- Multi-threshold assessment with neighbour-based evidence

---

## References

1. Rogers & Hahn, *J. Chem. Inf. Model.* 2010, 50(5), 742–754. (Morgan/ECFP fingerprints)
2. Sheridan, *J. Chem. Inf. Model.* 2012, 52(3), 814–823. (AD & uncertainty)
3. Baell & Holloway, *J. Med. Chem.* 2010, 53(7), 2719–2740. (PAINS)
4. Lipinski et al., *Adv. Drug Deliv. Rev.* 2001, 46(1–3), 3–26. (Ro5)
5. Veber et al., *J. Med. Chem.* 2002, 45(12), 2615–2623. (Veber rules)
6. Bickerton et al., *Nat. Chem.* 2012, 4(2), 90–98. (QED)
7. Hopkins et al., *Drug Discov. Today* 2004, 9(10), 430–431. (Ligand efficiency)
8. Wu et al., *Chem. Sci.* 2018, 9(2), 513–530. (Scaffold split / MoleculeNet)

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{chembLitz_cdk2,
  title  = {ChemBLitz: CDK2 Pharmacological Diagnostic Suite},
  author = {<Your Name>},
  year   = {2026},
  url    = {https://github.com/<your-username>/cdk2-ai-drug-discovery}
}
```

---

## Disclaimer

This is a **research decision-support tool** intended for hypothesis generation and compound prioritisation. Predictions require experimental validation and are not intended for clinical or safety-critical decisions.
