from uuid import uuid4

import numpy as np

from decision_platform.contracts import InteractionEvent
from decision_platform.data.events import LocalEventStore
from decision_platform.data.synthetic import generate_dataset
from decision_platform.experimentation.core import analyze, assign, sample_size
from decision_platform.features.store import point_in_time_counts
from decision_platform.policies.exploration import EpsilonGreedyPolicy
from decision_platform.retrieval.strategies import CollaborativeRetriever, PopularityRetriever


def test_synthetic_data_has_signal_and_valid_contracts() -> None:
    data = generate_dataset(40, 25, 500, seed=7)
    assert len(data.interactions) == 500
    assert {"impression", "click", "conversion"}.issuperset(set(data.interactions.event_type))
    assert data.interactions.label.max() == 3


def test_point_in_time_features_never_include_current_event() -> None:
    data = generate_dataset(10, 8, 100, seed=1)
    frame = point_in_time_counts(data.interactions)
    assert frame.groupby("user_id").first().user_prior_events.eq(0).all()


def test_retrievers_return_unique_top_k() -> None:
    data = generate_dataset(30, 20, 400, seed=2)
    for retriever in (PopularityRetriever(), CollaborativeRetriever(dimensions=5)):
        candidates = retriever.fit(data.interactions, data.items).retrieve("u00001", 10)
        assert len(candidates) == 10
        assert len({x.item_id for x in candidates}) == 10


def test_assignment_is_deterministic_and_balanced() -> None:
    assert assign("abc", "exp") == assign("abc", "exp")
    variants = [assign(str(i), "exp") for i in range(1000)]
    assert 0.45 < variants.count("treatment") / len(variants) < 0.55


def test_experiment_analysis_and_power() -> None:
    result = analyze(np.zeros(100), np.r_[np.ones(20), np.zeros(80)])
    assert result.absolute_effect == 0.2
    assert result.p_value < 0.01
    assert sample_size(0.1, 0.02) > 1000


def test_exploration_policy_is_reproducible_and_valid() -> None:
    policy = EpsilonGreedyPolicy(0.2, seed=5)
    first = policy.order(np.array([0.1, 0.8, 0.3]), 3)
    second = policy.order(np.array([0.1, 0.8, 0.3]), 3)
    assert np.array_equal(first[0], second[0])
    assert np.all((first[1] > 0) & (first[1] <= 1))


def test_event_store_is_idempotent(tmp_path) -> None:
    event = InteractionEvent(
        event_id=uuid4(), user_id="u", item_id="i", event_type="click", session_id="s"
    )
    store = LocalEventStore(tmp_path / "events.jsonl")
    assert store.put(event)
    assert not store.put(event)


def test_exploration_policy_rejects_negative_k() -> None:
    import pytest

    from decision_platform.policies.exploration import EpsilonGreedyPolicy

    with pytest.raises(ValueError):
        EpsilonGreedyPolicy().order(np.array([0.1, 0.2]), -1)


def test_population_stability_zero_for_identical_and_positive_for_shift() -> None:
    from decision_platform.monitoring.metrics import population_stability

    ref = np.linspace(0, 1, 500)
    assert population_stability(ref, ref) < 1e-6
    assert population_stability(ref, ref + 5) > 0.1


def test_recommendation_health_empty_returns_zeros() -> None:
    import pandas as pd

    from decision_platform.monitoring.metrics import recommendation_health

    catalog = pd.DataFrame({"item_id": [1], "category": ["a"], "popularity": [1.0]})
    result = recommendation_health(pd.DataFrame(columns=["request_id", "item_id"]), catalog)
    assert result == {"coverage": 0.0, "diversity": 0.0, "popularity_concentration": 0.0}
