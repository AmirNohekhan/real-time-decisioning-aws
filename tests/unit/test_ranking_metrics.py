import numpy as np
import pandas as pd
import pytest

from decision_platform.ranking.metrics import ranking_metrics


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"user_id": ["a", "a", "a", "b", "b"], "label": [0, 1, 0, 0, 0]})


def test_perfect_ranking_scores_one():
    metrics = ranking_metrics(_frame().iloc[:3], np.array([0.1, 0.9, 0.2]), k=3)
    assert metrics["ndcg_at_k"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["recall_at_k"] == pytest.approx(1.0)


def test_users_without_positives_score_zero_and_are_averaged():
    metrics = ranking_metrics(_frame(), np.array([0.1, 0.9, 0.2, 0.5, 0.4]), k=3)
    assert metrics["ndcg_at_k"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx(0.5)


def test_worse_ranking_lowers_mrr():
    metrics = ranking_metrics(_frame().iloc[:3], np.array([0.9, 0.1, 0.2]), k=3)
    assert metrics["mrr"] == pytest.approx(1 / 3)
