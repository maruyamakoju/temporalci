from __future__ import annotations

import math
from typing import Any


def to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def sprt_thresholds(*, alpha: float, beta: float) -> tuple[float, float] | None:
    if not (0.0 < alpha < 0.5):
        return None
    if not (0.0 < beta < 0.5):
        return None
    upper = math.log((1.0 - beta) / alpha)
    lower = math.log(beta / (1.0 - alpha))
    if not (math.isfinite(upper) and math.isfinite(lower)):
        return None
    return upper, lower


def drift_per_pair(*, effect_size: float, sigma: float) -> float | None:
    if effect_size <= 0.0 or sigma <= 0.0:
        return None
    drift = (effect_size * effect_size) / (2.0 * sigma * sigma)
    if not math.isfinite(drift) or drift <= 0.0:
        return None
    return drift


def required_pairs_from_thresholds(
    *,
    upper_threshold: float,
    lower_threshold: float,
    drift: float,
) -> tuple[float, float] | None:
    if drift <= 0.0:
        return None
    required_upper = upper_threshold / drift
    required_lower = abs(lower_threshold) / drift
    if not (math.isfinite(required_upper) and math.isfinite(required_lower)):
        return None
    return required_upper, required_lower


def estimate_required_pairs(
    *,
    alpha: float,
    beta: float,
    effect_size: float,
    sigma: float,
) -> dict[str, float] | None:
    thresholds = sprt_thresholds(alpha=alpha, beta=beta)
    if thresholds is None:
        return None
    upper_threshold, lower_threshold = thresholds

    drift = drift_per_pair(effect_size=effect_size, sigma=sigma)
    if drift is None:
        return None

    required_pairs = required_pairs_from_thresholds(
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        drift=drift,
    )
    if required_pairs is None:
        return None
    required_upper, required_lower = required_pairs
    return {
        "upper_threshold": upper_threshold,
        "lower_threshold": lower_threshold,
        "drift_per_pair": drift,
        "required_pairs_upper": required_upper,
        "required_pairs_lower": required_lower,
    }


def llr_per_pair(*, llr: float, paired_count: float) -> float | None:
    if paired_count <= 0.0:
        return None
    value = llr / paired_count
    if not math.isfinite(value):
        return None
    return value


def derive_sprt_metrics(
    *,
    effect_size: Any,
    sigma: Any,
    llr: Any,
    paired_count: Any,
    upper_threshold: Any | None = None,
    lower_threshold: Any | None = None,
    alpha: Any | None = None,
    beta: Any | None = None,
) -> dict[str, float | None]:
    effect_size_f = to_float(effect_size)
    sigma_f = to_float(sigma)
    llr_f = to_float(llr)
    paired_count_f = to_float(paired_count)
    upper_threshold_f = to_float(upper_threshold)
    lower_threshold_f = to_float(lower_threshold)
    alpha_f = to_float(alpha)
    beta_f = to_float(beta)

    if (upper_threshold_f is None or lower_threshold_f is None) and alpha_f is not None and beta_f is not None:
        thresholds = sprt_thresholds(alpha=alpha_f, beta=beta_f)
        if thresholds is not None:
            upper_threshold_f, lower_threshold_f = thresholds

    drift: float | None = None
    if effect_size_f is not None and sigma_f is not None:
        drift = drift_per_pair(effect_size=effect_size_f, sigma=sigma_f)

    required_pairs_upper: float | None = None
    required_pairs_lower: float | None = None
    if drift is not None and upper_threshold_f is not None and lower_threshold_f is not None:
        required_pairs = required_pairs_from_thresholds(
            upper_threshold=upper_threshold_f,
            lower_threshold=lower_threshold_f,
            drift=drift,
        )
        if required_pairs is not None:
            required_pairs_upper, required_pairs_lower = required_pairs

    llr_per_pair_value: float | None = None
    if llr_f is not None and paired_count_f is not None:
        llr_per_pair_value = llr_per_pair(llr=llr_f, paired_count=paired_count_f)

    return {
        "upper_threshold": upper_threshold_f,
        "lower_threshold": lower_threshold_f,
        "drift_per_pair": drift,
        "required_pairs_upper": required_pairs_upper,
        "required_pairs_lower": required_pairs_lower,
        "llr_per_pair": llr_per_pair_value,
    }
