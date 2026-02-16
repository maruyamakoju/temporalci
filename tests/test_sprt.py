from __future__ import annotations

import math

import pytest

from temporalci.sprt import derive_sprt_metrics
from temporalci.sprt import drift_per_pair
from temporalci.sprt import estimate_required_pairs
from temporalci.sprt import llr_per_pair
from temporalci.sprt import required_pairs_from_thresholds
from temporalci.sprt import sprt_thresholds


def test_sprt_thresholds_matches_formula() -> None:
    thresholds = sprt_thresholds(alpha=0.05, beta=0.1)
    assert thresholds is not None
    upper, lower = thresholds
    assert upper == pytest.approx(math.log((1.0 - 0.1) / 0.05))
    assert lower == pytest.approx(math.log(0.1 / (1.0 - 0.05)))


def test_estimate_required_pairs_matches_formula() -> None:
    payload = estimate_required_pairs(alpha=0.05, beta=0.1, effect_size=0.03, sigma=0.04)
    assert payload is not None
    drift = (0.03 * 0.03) / (2.0 * 0.04 * 0.04)
    upper = math.log((1.0 - 0.1) / 0.05)
    lower = math.log(0.1 / (1.0 - 0.05))
    assert payload["drift_per_pair"] == pytest.approx(drift)
    assert payload["upper_threshold"] == pytest.approx(upper)
    assert payload["lower_threshold"] == pytest.approx(lower)
    assert payload["required_pairs_upper"] == pytest.approx(upper / drift)
    assert payload["required_pairs_lower"] == pytest.approx(abs(lower) / drift)


def test_drift_and_required_pairs_validate_inputs() -> None:
    assert drift_per_pair(effect_size=0.02, sigma=0.0) is None
    assert required_pairs_from_thresholds(upper_threshold=1.0, lower_threshold=-1.0, drift=0.0) is None
    assert estimate_required_pairs(alpha=0.8, beta=0.1, effect_size=0.02, sigma=0.05) is None


def test_llr_per_pair_returns_none_when_pairs_zero() -> None:
    assert llr_per_pair(llr=3.0, paired_count=0.0) is None
    assert llr_per_pair(llr=3.0, paired_count=4.0) == pytest.approx(0.75)


def test_derive_sprt_metrics_uses_alpha_beta_when_thresholds_missing() -> None:
    derived = derive_sprt_metrics(
        effect_size=0.03,
        sigma=0.04,
        llr=3.2,
        paired_count=12,
        alpha=0.05,
        beta=0.1,
    )
    assert derived["upper_threshold"] is not None
    assert derived["lower_threshold"] is not None
    assert derived["drift_per_pair"] is not None
    assert derived["required_pairs_upper"] is not None
    assert derived["required_pairs_lower"] is not None
    assert derived["llr_per_pair"] == pytest.approx(3.2 / 12.0)


def test_derive_sprt_metrics_handles_missing_numbers() -> None:
    derived = derive_sprt_metrics(
        effect_size=None,
        sigma=None,
        llr="bad",
        paired_count=0,
        upper_threshold=None,
        lower_threshold=None,
    )
    assert derived["drift_per_pair"] is None
    assert derived["required_pairs_upper"] is None
    assert derived["required_pairs_lower"] is None
    assert derived["llr_per_pair"] is None
