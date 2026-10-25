import numpy as np
import pandas as pd

from decision_platform.monitoring.metrics import population_stability, recommendation_health


def test_population_stability_is_zero_for_identical_samples() -> None:
    sample = np.random.default_rng(0).normal(size=500)
    assert population_stability(sample, sample.copy()) < 1e-6


def test_population_stability_grows_with_shift() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(size=1000)
    small = population_stability(reference, rng.normal(0.2, 1.0, size=1000))
    large = population_stability(reference, rng.normal(3.0, 1.0, size=1000))
    assert 0.0 <= small < large


def test_recommendation_health_handles_empty_input() -> None:
    catalog = pd.DataFrame({"item_id": [1], "category": ["a"], "popularity": [1.0]})
    empty = pd.DataFrame({"request_id": [], "item_id": []})
    assert recommendation_health(empty, catalog) == {
        "coverage": 0.0,
        "diversity": 0.0,
        "popularity_concentration": 0.0,
    }


def test_recommendation_health_reports_coverage_and_diversity() -> None:
    catalog = pd.DataFrame(
        {"item_id": [1, 2, 3, 4], "category": ["a", "a", "b", "b"], "popularity": [1.0, 1.0, 1.0, 1.0]}
    )
    recs = pd.DataFrame({"request_id": ["r1", "r1"], "item_id": [1, 3]})
    health = recommendation_health(recs, catalog)
    assert health["coverage"] == 0.5
    assert health["diversity"] == 1.0
