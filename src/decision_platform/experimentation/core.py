import hashlib
from dataclasses import dataclass

import numpy as np
from scipy import stats


def assign(user_id: str, experiment_id: str, treatment_fraction: float = 0.5) -> str:
    if not 0 <= treatment_fraction <= 1:
        raise ValueError("treatment_fraction must be in [0, 1]")
    digest = hashlib.sha256(f"{experiment_id}:{user_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8]) / 2**64
    return "treatment" if bucket < treatment_fraction else "control"


@dataclass(frozen=True)
class ExperimentResult:
    control_rate: float
    treatment_rate: float
    absolute_effect: float
    relative_effect: float
    confidence_interval: tuple[float, float]
    p_value: float
    sample_ratio_mismatch_p_value: float


def analyze(control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05) -> ExperimentResult:
    if not len(control) or not len(treatment):
        raise ValueError("both variants need observations")
    p0, p1 = float(control.mean()), float(treatment.mean())
    effect = p1 - p0
    se = float(np.sqrt(p0 * (1 - p0) / len(control) + p1 * (1 - p1) / len(treatment)))
    z = stats.norm.ppf(1 - alpha / 2)
    pooled = (control.sum() + treatment.sum()) / (len(control) + len(treatment))
    pooled_se = np.sqrt(pooled * (1 - pooled) * (1 / len(control) + 1 / len(treatment)))
    p_value = 2 * stats.norm.sf(abs(effect / pooled_se)) if pooled_se else 1.0
    expected = (len(control) + len(treatment)) / 2
    srm = stats.chisquare([len(control), len(treatment)], [expected, expected]).pvalue
    return ExperimentResult(
        p0,
        p1,
        effect,
        effect / p0 if p0 else float("inf"),
        (effect - z * se, effect + z * se),
        float(p_value),
        float(srm),
    )


def sample_size(
    baseline: float, minimum_detectable_effect: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    target = baseline + minimum_detectable_effect
    if not 0 < baseline < 1 or not 0 < target < 1:
        raise ValueError("rates must be in (0, 1)")
    z_alpha, z_power = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pooled = (baseline + target) / 2
    numerator = (
        z_alpha * np.sqrt(2 * pooled * (1 - pooled))
        + z_power * np.sqrt(baseline * (1 - baseline) + target * (1 - target))
    ) ** 2
    return int(np.ceil(numerator / minimum_detectable_effect**2))
