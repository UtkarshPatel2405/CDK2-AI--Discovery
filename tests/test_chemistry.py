
import pytest
from rdkit import Chem


class TestMolFromSmiles:

    def test_valid_smiles(self):
        from src.chemistry import mol_from_smiles
        mol = mol_from_smiles("CCO")
        assert mol is not None
        assert mol.GetNumAtoms() == 3  # C, C, O (heavy atoms)

    def test_invalid_smiles(self):
        from src.chemistry import mol_from_smiles
        assert mol_from_smiles("INVALID_SMILES_XYZ") is None

    def test_empty_smiles(self):
        from src.chemistry import mol_from_smiles
        assert mol_from_smiles("") is None
        assert mol_from_smiles("  ") is None

    def test_none_input(self):
        from src.chemistry import mol_from_smiles
        # Should handle gracefully
        assert mol_from_smiles(None) is None

    def test_complex_smiles(self):
        from src.chemistry import mol_from_smiles
        smi = "CCC(CO)Nc1nc(NCc2ccccc2)c2ncn(C(C)C)c2n1"
        mol = mol_from_smiles(smi)
        assert mol is not None


class TestCanonicalSmiles:
    def test_canonicalization(self):
        from src.chemistry import mol_from_smiles, canonical_smiles
        mol = mol_from_smiles("C(O)C")
        can = canonical_smiles(mol)
        assert can == "CCO"

    def test_roundtrip(self):
        from src.chemistry import mol_from_smiles, canonical_smiles
        smi = "c1ccccc1"
        mol = mol_from_smiles(smi)
        can = canonical_smiles(mol)
        # Should be consistent
        mol2 = mol_from_smiles(can)
        can2 = canonical_smiles(mol2)
        assert can == can2


class TestInChIKey:
    def test_valid_molecule(self):
        from src.chemistry import mol_from_smiles, inchikey
        mol = mol_from_smiles("CCO")
        ik = inchikey(mol)
        assert ik != "N/A"
        assert len(ik) == 27 


class TestKeepLargestFragment:
    def test_salt_stripping(self):
        from src.chemistry import mol_from_smiles, keep_largest_fragment
        mol = mol_from_smiles("CCO.[Na+].[Cl-]")
        largest = keep_largest_fragment(mol)
        assert largest.GetNumHeavyAtoms() == 3

    def test_single_fragment(self):
        from src.chemistry import mol_from_smiles, keep_largest_fragment
        mol = mol_from_smiles("CCO")
        largest = keep_largest_fragment(mol)
        assert largest.GetNumHeavyAtoms() == 3


class TestMolToPng:
    def test_renders_to_bytes(self):
        from src.chemistry import mol_from_smiles, mol_to_png_bytes
        mol = mol_from_smiles("CCO")
        png = mol_to_png_bytes(mol)
        assert isinstance(png, bytes)
        assert len(png) > 0
        # PNG magic bytes
        assert png[:4] == b"\x89PNG"
