import numpy as np
import pytest
from rdkit import Chem


def _mol(smi: str):
    return Chem.MolFromSmiles(smi)


class TestMorganFingerprint:
    def test_output_shape(self):
        from src.descriptors import fp_to_array, morgan_fp
        mol = _mol("CCO")
        fp = morgan_fp(mol)
        arr = fp_to_array(fp)
        assert arr.shape == (2048,)
        assert arr.dtype == np.int8

    def test_different_molecules_differ(self):
        from src.descriptors import fp_to_array, morgan_fp
        fp1 = fp_to_array(morgan_fp(_mol("CCO")))
        fp2 = fp_to_array(morgan_fp(_mol("c1ccccc1")))
        assert not np.array_equal(fp1, fp2)

    def test_same_molecule_same_fp(self):
        from src.descriptors import fp_to_array, morgan_fp
        fp1 = fp_to_array(morgan_fp(_mol("CCO")))
        fp2 = fp_to_array(morgan_fp(_mol("OCC")))
        assert np.array_equal(fp1, fp2)


class TestPhysicoChemProfile:
    def test_ethanol(self):
        from src.descriptors import compute_physicochemical
        p = compute_physicochemical(_mol("CCO"))
        assert 40 < p.mol_wt < 50
        assert p.hbd >= 1
        assert p.num_heavy_atoms == 3

    def test_aspirin(self):
        from src.descriptors import compute_physicochemical
        p = compute_physicochemical(_mol("CC(=O)Oc1ccccc1C(=O)O"))
        assert 170 < p.mol_wt < 190
        assert p.num_rings == 1


class TestRo5:
    def test_drug_like_molecule(self):
        from src.descriptors import ro5_violations
        assert ro5_violations(_mol("CC(=O)Oc1ccccc1C(=O)O")) == 0

    def test_high_mw_violation(self):
        from src.descriptors import ro5_violations
        big = _mol("C" * 50) 
        v = ro5_violations(big)
        assert v >= 1  


class TestVeber:
    def test_small_drug(self):
        from src.descriptors import veber_pass
        assert veber_pass(_mol("CCO")) is True


class TestLigandEfficiency:
    def test_positive_pic50(self):
        from src.descriptors import calculate_ligand_efficiency
        le = calculate_ligand_efficiency(7.0, _mol("c1ccccc1"))
        assert le > 0

    def test_zero_heavy_atoms(self):
        from src.descriptors import calculate_ligand_efficiency
        mol = Chem.MolFromSmiles("[H][H]")
        if mol is not None:
            le = calculate_ligand_efficiency(7.0, mol)


class TestPotencyBand:
    def test_strong(self):
        from src.descriptors import potency_band
        assert potency_band(8.5) == "Strong"

    def test_moderate(self):
        from src.descriptors import potency_band
        assert potency_band(6.5) == "Moderate"

    def test_weak(self):
        from src.descriptors import potency_band
        assert potency_band(4.0) == "Weak"


class TestPic50Conversion:
    def test_conversion(self):
        from src.descriptors import pic50_to_ic50_nM
        assert abs(pic50_to_ic50_nM(9.0) - 1.0) < 0.01
        assert abs(pic50_to_ic50_nM(6.0) - 1000.0) < 0.1

    def test_monotonic(self):
        from src.descriptors import pic50_to_ic50_nM
        assert pic50_to_ic50_nM(8.0) < pic50_to_ic50_nM(6.0)
