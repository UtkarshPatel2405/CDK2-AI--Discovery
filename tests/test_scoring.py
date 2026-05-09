import pytest
from src.scoring import make_priority
from src.similarity import ad_band


class TestMakePriority:
    def test_high_priority(self):
        result = make_priority(pred_pic50=7.5, pred_std=0.3, max_sim=0.5, pains_hits=[])
        assert result == "High"

    def test_low_priority_weak_compound(self):
        result = make_priority(pred_pic50=4.0, pred_std=0.9, max_sim=0.1, pains_hits=[])
        assert result == "Low"

    def test_pains_penalizes(self):
        result = make_priority(pred_pic50=7.5, pred_std=0.3, max_sim=0.5, pains_hits=["PAINS_A"])
        assert result in ("Medium", "Low")


class TestAdBand:
    def test_in_domain(self):
        assert ad_band(0.60) == "In-domain"

    def test_borderline(self):
        assert ad_band(0.40) == "Borderline"

    def test_out_of_domain(self):
        assert ad_band(0.20) == "Out-of-domain risk"

    def test_exact_boundary(self):
        assert ad_band(0.50) == "In-domain"
        assert ad_band(0.30) == "Borderline"
