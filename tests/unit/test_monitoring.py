import numpy as np
import pandas as pd

from decision_platform.monitoring.metrics import population_stability, recommendation_health


def test_recommendation_health_empty_recommendations_returns_zeros() -> None:
    catalog = pd.DataFrame({"item_id": [1, 2], "category": ["a", "b"], "popularity": [1, 2]})
    recommendations = pd.DataFrame(columns=["item_id", "request_id"])

    result = recommendation_health(recommendations, catalog)

    assert result == {"coverage": 0.0, "diversity": 0.0, "popularity_concentration": 0.0}


def test_recommendation_health_computes_expected_metrics() -> None:
    catalog = pd.DataFrame(
        {
            "item_id": [1, 2, 3, 4],
            "category": ["a", "a", "b", "b"],
            "popularity": [10, 20, 30, 40],
        }
    )
    recommendations = pd.DataFrame(
        {
            "request_id": [1, 1, 2, 2],
            "item_id": [1, 2, 3, 4],
        }
    )

    result = recommendation_health(recommendations, catalog)

    assert result["coverage"] == 1.0
    assert result["diversity"] == 0.5
    assert result["popularity_concentration"] == 1.0


def test_population_stability_identical_distributions_near_zero() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(size=1000)

    stat = population_stability(reference, reference.copy())

    assert stat < 1e-6


def test_population_stability_detects_distribution_shift() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(loc=0, size=1000)
    similar = rng.normal(loc=0, size=1000)
    shifted = rng.normal(loc=5, size=1000)

    stat_similar = population_stability(reference, similar)
    stat_shifted = population_stability(reference, shifted)

    assert stat_shifted > stat_similar
