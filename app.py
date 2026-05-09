from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.chemistry import canonical_smiles, inchikey, keep_largest_fragment, mol_from_smiles, mol_to_png_bytes
from src.config import (
    DEFAULT_TRIAGE_PIC50, DEFAULT_TRIAGE_SIM, DEFAULT_TRIAGE_STD,
    FP_NBITS, FP_RADIUS, LEGACY_DATA_PATH, LEGACY_MODEL_PATH, MODEL_DRIVE_FILE_ID,
)
from src.data import (
    _chembl_molecule_url, add_scaffold_column, chembl_pref_name,
    create_pains_catalog, download_model_from_drive, load_dataset, scaffold_stats,
)
from src.descriptors import (
    calculate_ligand_efficiency, check_pains, compute_physicochemical,
    morgan_fp, pic50_to_ic50_nM, potency_band, ro5_violations,
    uncertainty_band, veber_pass,
)
from src.model import rf_predict
from src.scoring import BatchRow, decision_text, make_priority, score_smiles_row
from src.similarity import ad_band, build_dataset_fps, compute_similarity_and_neighbors

st.set_page_config(page_title="ChemBLitz Professional | CDK2 Suite", layout="wide")

st.markdown(
"""
<style>
html, body {
  font-family: "Times New Roman", Times, serif !important;
  font-size: 18px !important;
  line-height: 1.45 !important;
}
.stMarkdown, .stMarkdown p, .stMarkdown li {
  font-family: "Times New Roman", Times, serif !important;
  font-size: 18px !important;
  line-height: 1.55 !important;
}
button[data-baseweb="tab"] { font-size: 16px !important; }
div[data-testid="stMetricLabel"] p {
  font-size: 16px !important;
  font-weight: 700 !important;
  color: #1f77b4 !important;
  line-height: 1.2 !important;
}
div[data-testid="stMetricValue"] {
  font-size: 26px !important;
  font-weight: 600 !important;
  line-height: 1.1 !important;
}
div[data-testid="stExpander"] summary {
  font-size: 16px !important;
  line-height: 1.25 !important;
}
.stButton > button {
  font-size: 16px !important;
  border-radius: 8px !important;
  padding: 0.65em 1.1em !important;
}
.block-container { padding-top: 1.1rem !important; }
@media (min-width: 1100px) {
  .block-container { max-width: 1500px !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_resource
def _load_assets():
    if not LEGACY_MODEL_PATH.exists():
        download_model_from_drive()
    import joblib
    model = joblib.load(LEGACY_MODEL_PATH)
    df = load_dataset()
    return model, df

@st.cache_resource
def _get_pains():
    return create_pains_catalog()

@st.cache_resource
def _scaffold_col(df: pd.DataFrame):
    return add_scaffold_column(df)

@st.cache_resource
def _cached_fps(df: pd.DataFrame):
    return build_dataset_fps(df["smiles"])

st.title("CDK2 Pharmacological Diagnostic Suite")
st.caption("Prediction + Evidence + Interpretation for CDK2 inhibitor discovery")

try:
    model, df = _load_assets()
    pains_catalog = _get_pains()
except Exception as e:
    st.error(f"Critical System Failure: {e}")
    st.stop()

df_scaf = _scaffold_col(df)

tab_report, tab_library, tab_space, tab_method = st.tabs(
    ["Single Compound Report", "Library Triage (CSV)", "Chemical Space", "Methodology / Model Card"]
)

with tab_report:
    top_left, top_right = st.columns([2.3, 1], gap="large")

    with top_left:
        st.subheader("Input & evidence settings")
        c1, c2, c3 = st.columns([1.2, 1.0, 1.0], gap="large")

        top_binders = df.sort_values("pic50", ascending=False).head(5)
        refs = {
            f"{r.get('molecule_chembl_id','')} (pIC50 {r['pic50']:.1f})": r["smiles"]
            for _, r in top_binders.iterrows()
        }

        with c1:
            selected_ref = st.selectbox("Reference ligand", ["None"] + list(refs.keys()))
            if selected_ref != "None":
                st.session_state["query_smiles"] = refs[selected_ref]
            target_smiles = st.text_input("SMILES", value=st.session_state.get("query_smiles", ""))

        with c2:
            p_min, p_max = float(df["pic50"].min()), float(df["pic50"].max())
            pic_range = st.slider("Evidence pIC50 range", p_min, p_max, (p_min, p_max))
            min_meas = st.slider("Min measurements", 1, int(df["n_measurements"].max()), 1)

        with c3:
            strip_salts = st.checkbox("Keep largest fragment (salt stripping)", value=True)
            compute_ad = st.checkbox("Compute AD + neighbors", value=True)
            topk_neighbors = st.slider("Top-K neighbors", 5, 25, 10)

        df_f = df[
            (df["pic50"] >= pic_range[0]) & (df["pic50"] <= pic_range[1]) & (df["n_measurements"] >= min_meas)
        ].copy()
        st.info(f"Evidence subset size: {len(df_f)} compounds (from {len(df)} total)")

        with st.expander("Interpretation guide (what do these numbers mean?)", expanded=False):
            st.markdown("""
- **pIC50**: -log10(IC50 in molar). +1 pIC50  10× stronger potency.
- **IC50 (nM)**: 10^(9 - pIC50). Reported for intuition.
- **σ (uncertainty)**: std across RF trees (model disagreement). Not a calibrated CI.
- **Applicability Domain (AD)**: similarity to training chemistry. Low similarity = extrapolation risk.
- **Ligand Efficiency (LE)**: potency normalized by size (heavy atom count). Useful for lead-likeness.
- **PAINS**: risk flags for assay interference (requires orthogonal validation).
- **Scaffold evidence**: how common the Murcko scaffold is + activity distribution in dataset.
""")

    with top_right:
        st.subheader("Run")
        run = st.button("Explore / Execute", type="primary", use_container_width=True)
        st.caption("Tip: Keep AD enabled for professional reliability assessment.")

    st.divider()

    rep_left, rep_right = st.columns([1, 1], gap="large")

    if not run:
        st.info("Set inputs above and click **Explore / Execute** to generate a report.")
    else:
        mol = mol_from_smiles(target_smiles)
        if mol is None:
            st.error("Invalid structure: SMILES parsing failed.")
        else:
            if strip_salts:
                mol = keep_largest_fragment(mol)

            smiles_can = canonical_smiles(mol)
            ik = inchikey(mol)
            pred_pic50, pred_std, fp = rf_predict(model, mol)
            le = calculate_ligand_efficiency(pred_pic50, mol)
            pains_hits = check_pains(mol, pains_catalog)
            phys = compute_physicochemical(mol)

            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smi = Chem.MolToSmiles(scaffold_mol) if scaffold_mol else ""

            match = df[df["inchikey"].astype(str) == ik]
            in_dataset = len(match) > 0
            exp_pic50 = float(match["pic50"].median()) if in_dataset else np.nan
            exp_n = int(match["n_measurements"].max()) if in_dataset else 0
            exp_chembl = str(match.iloc[0]["molecule_chembl_id"]).strip() if in_dataset else ""
            exp_name = chembl_pref_name(exp_chembl) if exp_chembl else None

            max_sim = mean_sim = 0.0
            n05 = n04 = n03 = 0
            neighbors = pd.DataFrame()
            if compute_ad:
                fps, idx_map = _cached_fps(df_f)
                max_sim, mean_sim, n05, n04, n03, neighbors = compute_similarity_and_neighbors(
                    fp, fps, idx_map, df_f, topk_neighbors
                )

            priority = make_priority(pred_pic50, pred_std, max_sim if compute_ad else 0.0, pains_hits)
            rationale, next_steps = decision_text(pred_pic50, pred_std, max_sim if compute_ad else 0.0, le, pains_hits)

            sc_count, sc_min, sc_med, sc_max, sc_top = 0, np.nan, np.nan, np.nan, pd.DataFrame()
            if scaffold_smi:
                sc_count, sc_min, sc_med, sc_max, sc_top = scaffold_stats(df_scaf, scaffold_smi)

            with rep_left:
                with st.container(border=True):
                    st.markdown("## Executive summary")
                    if priority == "High":
                        st.success(f"Priority: **{priority}**")
                    elif priority == "Medium":
                        st.warning(f"Priority: **{priority}**")
                    else:
                        st.info(f"Priority: **{priority}**")
                    st.markdown("**Rationale**")
                    for r in rationale:
                        st.write(f"- {r}")
                    st.markdown("**Next steps**")
                    for s in next_steps:
                        st.write(f"- {s}")

                with st.container(border=True):
                    st.markdown("## Predicted potency & efficiency")
                    img_col, met_col = st.columns([1.15, 1.0], gap="large")
                    with img_col:
                        st.image(mol_to_png_bytes(mol), caption="Query structure (RDKit 2D)")
                    with met_col:
                        m1, m2 = st.columns(2)
                        m1.metric("Pred pIC50", f"{pred_pic50:.2f}")
                        m2.metric("Pred IC50 (nM)", f"{pic50_to_ic50_nM(pred_pic50):.1f}")
                        m3, m4 = st.columns(2)
                        m3.metric("σ (trees)", f"{pred_std:.3f}")
                        m4.metric("LE", f"{le:.2f}")
                        st.caption(f"Potency band: **{potency_band(pred_pic50)}**")
                        st.caption(f"Uncertainty band: **{uncertainty_band(pred_std)}**")

                with st.container(border=True):
                    st.markdown("## Identifiers & dataset match")
                    st.write("Canonical SMILES:")
                    st.code(smiles_can, language="text")
                    st.write("InChIKey:", ik)
                    if in_dataset:
                        st.success(f"Exact match in dataset: {exp_chembl} (n_measurements={exp_n})")
                        if exp_name:
                            st.write("ChEMBL pref_name:", exp_name)
                        st.write("ChEMBL URL:", _chembl_molecule_url(exp_chembl))
                        st.write(f"Experimental pIC50 (median): {exp_pic50:.2f}")
                        st.write(f"ΔpIC50 (Pred  Exp): {(pred_pic50 - exp_pic50):+.2f}")
                    else:
                        st.warning("No exact match in dataset (by InChIKey).")

            with rep_right:
                with st.container(border=True):
                    st.markdown("## Evidence & applicability domain")
                    if compute_ad:
                        a1, a2, a3, a4, a5 = st.columns(5)
                        a1.metric("Max sim", f"{max_sim:.3f}")
                        a2.metric("Mean sim", f"{mean_sim:.3f}")
                        a3.metric("0.50", str(n05))
                        a4.metric("0.40", str(n04))
                        a5.metric("0.30", str(n03))
                        band = ad_band(max_sim)
                        if band == "In-domain":
                            st.success(band)
                        elif band == "Borderline":
                            st.warning(band)
                        else:
                            st.error(band)
                        if len(neighbors):
                            pic = neighbors["pic50_exp"].dropna()
                            if len(pic):
                                st.caption(
                                    f"Neighbor pIC50 spread (Top-{min(len(neighbors), topk_neighbors)}): "
                                    f"min={pic.min():.2f}, median={pic.median():.2f}, max={pic.max():.2f}"
                                )
                            with st.expander("Top nearest neighbors (experimental evidence)", expanded=False):
                                st.dataframe(neighbors, use_container_width=True, hide_index=True)
                    else:
                        st.info("AD disabled. Enable it above for evidence-based confidence.")

                with st.container(border=True):
                    st.markdown("## Risk & developability")
                    if pains_hits:
                        st.error(f"PAINS alerts: {', '.join(pains_hits)}")
                    else:
                        st.success("No PAINS alerts detected.")
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("MolWt", f"{phys.mol_wt:.1f}")
                    d2.metric("cLogP", f"{phys.clogp:.2f}")
                    d3.metric("TPSA", f"{phys.tpsa:.1f}")
                    d4.metric("QED", f"{phys.qed:.2f}")
                    f1, f2, f3 = st.columns(3)
                    f1.write(f"**Ro5 violations:** {ro5_violations(mol)}")
                    f2.write(f"**Veber:** {'PASS' if veber_pass(mol) else 'FAIL'}")
                    f3.write(f"**Rings:** {phys.num_rings}")
                    g1, g2, g3 = st.columns(3)
                    g1.write(f"**HBD/HBA:** {phys.hbd}/{phys.hba}")
                    g2.write(f"**RotB:** {phys.rotatable_bonds}")
                    g3.write(f"**Size risk:** {'High' if phys.mol_wt > 500 else 'OK'}")

                with st.container(border=True):
                    st.markdown("## Series context (scaffold)")
                    if scaffold_smi:
                        st.write("Murcko scaffold:")
                        st.code(scaffold_smi, language="text")
                        st.write(f"Occurrences in dataset: **{sc_count}**")
                        if sc_count > 0:
                            st.write(f"Scaffold pIC50: min={sc_min:.2f}, median={sc_med:.2f}, max={sc_max:.2f}")
                            with st.expander("Top scaffold exemplars", expanded=False):
                                st.dataframe(sc_top, use_container_width=True, hide_index=True)
                        else:
                            st.info("Scaffold not found in dataset (new chemotype).")
                    else:
                        st.info("No scaffold computed for this structure.")

with tab_library:
    st.subheader("Library triage (CSV)")
    st.write("Upload CSV with `smiles` column (optional `id`). Outputs pass/fail triage + export.")

    cA, cB, cC = st.columns([1.2, 1.2, 1.2], gap="large")
    with cA:
        triage_pic50 = st.number_input("Min predicted pIC50", value=float(DEFAULT_TRIAGE_PIC50), step=0.1)
    with cB:
        triage_std = st.number_input("Max σ (uncertainty)", value=float(DEFAULT_TRIAGE_STD), step=0.05)
    with cC:
        triage_sim = st.number_input("Min max similarity", value=float(DEFAULT_TRIAGE_SIM), step=0.05)

    p_min, p_max = float(df["pic50"].min()), float(df["pic50"].max())
    pic_range = st.slider("Evidence pIC50 range", p_min, p_max, (p_min, p_max), key="lib_pic")
    min_meas = st.slider("Min measurements", 1, int(df["n_measurements"].max()), 1, key="lib_meas")
    strip_salts_lib = st.checkbox("Keep largest fragment (salt stripping)", value=True, key="lib_salt")
    compute_ad_lib = st.checkbox("Compute AD (recommended)", value=True, key="lib_ad")
    topk_lib = st.slider("Top-K neighbors (for AD counts only)", 5, 25, 10, key="lib_k")

    df_f_lib = df[
        (df["pic50"] >= pic_range[0]) & (df["pic50"] <= pic_range[1]) & (df["n_measurements"] >= min_meas)
    ].copy()
    st.info(f"Evidence subset size: {len(df_f_lib)} compounds")

    fps_lib, idx_lib = _cached_fps(df_f_lib)

    file = st.file_uploader("Upload CSV", type=["csv"], key="lib_csv")
    if file is not None:
        try:
            batch = pd.read_csv(file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        if "smiles" not in batch.columns:
            st.error("CSV must include `smiles` column.")
            st.stop()

        if "id" not in batch.columns:
            batch = batch.copy()
            batch["id"] = [f"row_{i+1}" for i in range(len(batch))]

        run_batch = st.button("Run library triage", type="primary")
        if run_batch:
            rows = []
            with st.spinner("Scoring library..."):
                for rid, smi in zip(batch["id"].astype(str).tolist(), batch["smiles"].astype(str).tolist()):
                    r = score_smiles_row(
                        smi, rid, model=model, pains_catalog=pains_catalog,
                        df_evidence=df_f_lib, df_full=df, df_scaf=df_scaf,
                        dataset_fps=fps_lib, dataset_idx_map=idx_lib,
                        strip_salts=strip_salts_lib, compute_ad_flag=compute_ad_lib,
                        topk_neighbors=topk_lib,
                        triage_pic50=float(triage_pic50),
                        triage_std=float(triage_std),
                        triage_sim=float(triage_sim),
                    )
                    rows.append(asdict(r))

            out = pd.DataFrame(rows)
            st.success("Library triage complete.")
            st.write("Pass triage:", f"{int(out['pass_triage'].sum())} / {len(out)}")
            st.dataframe(out, use_container_width=True, hide_index=True)

            st.download_button(
                "Download full results CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="cdk2_library_triage_results.csv",
                mime="text/csv",
            )
            top = out[out["pass_triage"] == True].copy()
            if len(top) > 0:
                st.download_button(
                    "Download PASS-TRIAGE only CSV",
                    data=top.to_csv(index=False).encode("utf-8"),
                    file_name="cdk2_library_pass_triage.csv",
                    mime="text/csv",
                )

with tab_space:
    st.subheader("Chemical space")
    st.write("Explore experimental dataset distribution and evidence subset context.")

    p_min, p_max = float(df["pic50"].min()), float(df["pic50"].max())
    pic_range = st.slider("pIC50 range (plot)", p_min, p_max, (p_min, p_max), key="space_pic")
    min_meas = st.slider("Min measurements (plot)", 1, int(df["n_measurements"].max()), 1, key="space_meas")
    df_space = df[(df["pic50"] >= pic_range[0]) & (df["pic50"] <= pic_range[1]) & (df["n_measurements"] >= min_meas)].copy()

    if "ic50_nM" in df_space.columns:
        fig = px.scatter(
            df_space, x="pic50", y="ic50_nM", size="n_measurements",
            hover_name="molecule_chembl_id", template="plotly_white",
            labels={"pic50": "Experimental pIC50", "ic50_nM": "IC50 (nM)"},
        )
    else:
        fig = px.scatter(
            df_space, x="pic50", y="n_measurements", size="n_measurements",
            hover_name="molecule_chembl_id", template="plotly_white",
            labels={"pic50": "Experimental pIC50", "n_measurements": "n_measurements"},
        )
    st.plotly_chart(fig, use_container_width=True)

with tab_method:
    st.subheader("Model card / interpretation")
    st.markdown(
        f"""
### Model
- **Task:** CDK2 pIC50 regression
- **Model:** RandomForest on Morgan fingerprints (radius={FP_RADIUS}, {FP_NBITS} bits)
- **Uncertainty:** σ across RF trees (disagreement proxy)
- **Training data:** Curated CDK2 IC50 data from ChEMBL (CHEMBL301)
- **Split strategy:** Scaffold-based split (Murcko scaffolds)
- **Validation:** 5-fold cross-validation + Y-randomisation test

### Evidence / Applicability domain (AD)
- Uses Tanimoto similarity to the selected evidence subset.
- Rule of thumb:
  - max sim  0.50: in-domain
  - 0.30–0.50: borderline
  - < 0.30: out-of-domain risk

### Why two-column report?
Left: decision + potency + identifiers (what you decide).
Right: evidence + risk + scaffold context (why you trust it).

### References
- Rogers & Hahn, J. Chem. Inf. Model. 2010, 50(5), 742-754 (Morgan/ECFP)
- Sheridan, J. Chem. Inf. Model. 2012, 52(3), 814-823 (AD & uncertainty)
- Baell & Holloway, J. Med. Chem. 2010, 53(7), 2719-2740 (PAINS)
- Lipinski et al., Adv. Drug Deliv. Rev. 2001, 46, 3-26 (Ro5)
- Bickerton et al., Nat. Chem. 2012, 4(2), 90-98 (QED)

### Intended use
Decision support for hit prioritization and analog ranking. Requires experimental validation.
"""
    )
