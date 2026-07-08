"""Unit tests for the Geo & Resilience Intelligence Engine."""

import pytest
from ai.engines.geo_resilience import GeoResilienceEngine
from ai.schemas.input_schema import BorrowerInput, EmploymentType, GeoInput, HomeOwnership
from ai.utils.geo_data import get_state_risk_profile, get_composite_climate_risk, list_all_states


def make_geo(state: str = "Maharashtra", **kwargs) -> GeoInput:
    return GeoInput(state=state, **kwargs)


def make_borrower() -> BorrowerInput:
    return BorrowerInput(
        age=35, annual_income=800_000,
        employment_type=EmploymentType.SALARIED,
        employment_length_months=48,
        home_ownership=HomeOwnership.RENT,
    )


class TestGeoResilienceEngine:
    def setup_method(self):
        self.engine = GeoResilienceEngine()

    def test_compute_returns_output(self):
        result = self.engine.compute(make_geo(), make_borrower())
        assert result is not None

    def test_resilience_score_in_range(self):
        result = self.engine.compute(make_geo(), make_borrower())
        assert 0 <= result.geo_resilience_score <= 100

    def test_all_risk_scores_in_range(self):
        result = self.engine.compute(make_geo(), make_borrower())
        for attr in ("flood_risk", "drought_risk", "cyclone_risk", "heatwave_risk"):
            val = getattr(result, attr)
            assert 0 <= val <= 1, f"{attr} = {val} out of range"

    def test_composite_risk_in_range(self):
        result = self.engine.compute(make_geo(), make_borrower())
        assert 0 <= result.composite_climate_risk <= 1

    def test_risk_adjustment_positive(self):
        result = self.engine.compute(make_geo(), make_borrower())
        assert result.risk_adjustment > 0

    def test_high_risk_state_lower_resilience(self):
        assam = self.engine.compute(make_geo("Assam"), make_borrower())
        delhi = self.engine.compute(make_geo("Delhi"), make_borrower())
        assert assam.geo_resilience_score < delhi.geo_resilience_score

    def test_unknown_state_uses_default(self):
        result = self.engine.compute(make_geo("UnknownState123"), make_borrower())
        assert result is not None
        assert 0 <= result.geo_resilience_score <= 100

    def test_data_source_is_offline(self):
        result = self.engine.compute(make_geo(), make_borrower())
        assert result.data_source == "STATIC_OFFLINE"

    def test_climate_impact_valid_enum(self):
        result = self.engine.compute(make_geo(), make_borrower())
        assert result.climate_impact.value in {"LOW", "MODERATE", "HIGH", "SEVERE"}


class TestGeoDataLookup:
    def test_all_states_return_profile(self):
        for state in list_all_states():
            profile = get_state_risk_profile(state)
            assert profile is not None

    def test_assam_high_flood_risk(self):
        profile = get_state_risk_profile("Assam")
        assert profile.flood_risk >= 0.8

    def test_rajasthan_high_drought_risk(self):
        profile = get_state_risk_profile("Rajasthan")
        assert profile.drought_risk >= 0.8

    def test_odisha_high_cyclone_risk(self):
        profile = get_state_risk_profile("Odisha")
        assert profile.cyclone_risk >= 0.7

    def test_case_insensitive_lookup(self):
        p1 = get_state_risk_profile("maharashtra")
        p2 = get_state_risk_profile("Maharashtra")
        assert p1.state == p2.state or p1.flood_risk == p2.flood_risk
