import numpy as np
import pandas as pd
import pytest

from decision_platform.ranking.metrics import ranking_metrics


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"user_id": ["a", "a", "a"], "label": [0, 1, 3]})


def test_perfect_ordering_scores_one() -> None:
    metrics = ranking_metrics(_frame(), np.array([0.1, 0.5, 0.9]), k=3)
    assert metrics["ndcg_at_k"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)


def test_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        ranking_metrics(_frame(), np.zeros(3), k=0)


def test_empty_frame_returns_zeros() -> None:
    empty = pd.DataFrame({"user_id": [], "label": []})
    metrics = ranking_metrics(empty, np.array([]))
    assert set(metrics.values()) == {0.0}
