import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from decision_platform.retrieval.base import Candidate


class PopularityRetriever:
    name = "popularity"

    def __init__(self) -> None:
        self._ranked: list[Candidate] = []

    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame) -> "PopularityRetriever":
        weights = interactions.event_type.map(
            {"impression": 0.1, "click": 1.0, "save": 2.0, "conversion": 4.0}
        )
        scores = weights.groupby(interactions.item_id).sum().reindex(items.item_id, fill_value=0)
        normalized = scores / max(float(scores.max()), 1.0)
        self._ranked = [
            Candidate(str(i), float(s), self.name)
            for i, s in normalized.sort_values(ascending=False).items()
        ]
        return self

    def retrieve(self, user_id: str, k: int) -> list[Candidate]:
        del user_id
        return self._ranked[:k]


class CollaborativeRetriever:
    name = "collaborative_svd"

    def __init__(self, dimensions: int = 16, seed: int = 42) -> None:
        self.dimensions = dimensions
        self.seed = seed
        self._users: dict[str, int] = {}
        self._items: list[str] = []
        self._scores: np.ndarray | None = None
        self._fallback: list[Candidate] = []

    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame) -> "CollaborativeRetriever":
        frame = interactions.copy()
        frame["weight"] = frame.event_type.map(
            {"impression": 0.0, "click": 1.0, "save": 2.0, "conversion": 4.0}
        )
        matrix = frame.pivot_table(
            index="user_id", columns="item_id", values="weight", aggfunc="sum", fill_value=0
        )
        self._users = {str(u): i for i, u in enumerate(matrix.index)}
        self._items = [str(i) for i in matrix.columns]
        n_components = max(1, min(self.dimensions, min(matrix.shape) - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
        user_factors = svd.fit_transform(matrix.to_numpy())
        self._scores = user_factors @ svd.components_
        popularity = PopularityRetriever().fit(interactions, items)
        self._fallback = popularity.retrieve("", len(items))
        return self

    def retrieve(self, user_id: str, k: int) -> list[Candidate]:
        if self._scores is None or user_id not in self._users:
            return self._fallback[:k]
        row = self._scores[self._users[user_id]]
        indices = np.argsort(-row)[:k]
        scale = max(float(np.max(np.abs(row))), 1e-9)
        return [Candidate(self._items[i], float(row[i] / scale), self.name) for i in indices]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, retrievers: list[object] | None = None) -> None:
        self.retrievers = retrievers or [PopularityRetriever(), CollaborativeRetriever()]

    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame) -> "HybridRetriever":
        for retriever in self.retrievers:
            retriever.fit(interactions, items)  # type: ignore[attr-defined]
        return self

    def retrieve(self, user_id: str, k: int) -> list[Candidate]:
        merged: dict[str, Candidate] = {}
        for retriever in self.retrievers:
            for candidate in retriever.retrieve(user_id, k):  # type: ignore[attr-defined]
                old = merged.get(candidate.item_id)
                score = candidate.score if old is None else max(old.score, candidate.score)
                source = candidate.source if old is None else f"{old.source}+{candidate.source}"
                merged[candidate.item_id] = Candidate(candidate.item_id, score, source)
        return sorted(merged.values(), key=lambda x: (-x.score, x.item_id))[:k]


def recall_at_k(retriever: object, holdout: pd.DataFrame, k: int = 50) -> float:
    relevant = holdout[holdout.label > 0].groupby("user_id").item_id.apply(set)
    if relevant.empty:
        return 0.0
    values = []
    for user_id, truth in relevant.items():
        found = {c.item_id for c in retriever.retrieve(str(user_id), k)}  # type: ignore[attr-defined]
        values.append(len(found & truth) / len(truth))
    return float(np.mean(values))
