from datetime import UTC, datetime

import numpy as np

from decision_platform.contracts import (
    ComponentScores,
    Recommendation,
    RecommendationRequest,
    RecommendationResponse,
)
from decision_platform.experimentation.core import assign
from decision_platform.features.store import FeatureStore
from decision_platform.policies.exploration import EpsilonGreedyPolicy
from decision_platform.ranking.features import build_inference_frame
from decision_platform.ranking.model import RankingModel, propensity
from decision_platform.retrieval.base import RetrievalService


class DecisionEngine:
    def __init__(
        self,
        store: FeatureStore,
        retriever: RetrievalService,
        ranker: RankingModel,
        epsilon: float = 0.05,
    ) -> None:
        self.store, self.retriever, self.ranker, self.epsilon = store, retriever, ranker, epsilon

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        experiment = request.experiment_id or "none"
        assignment = assign(request.user_id, experiment) if experiment != "none" else "control"
        candidates = self.retriever.retrieve(request.user_id, max(request.k * 5, 50))
        retrieval = {c.item_id: c.score for c in candidates}
        sources = {c.item_id: c.source for c in candidates}
        items = self.store.get_items(list(retrieval))
        items = items[items.available.astype(bool)].copy()
        user = dict(self.store.get_user(request.user_id))
        frame = build_inference_frame(user, items, request.context.model_dump(), retrieval)
        relevance = self.ranker.predict(frame)
        conversion = propensity(relevance)
        value = conversion * np.log1p(frame.price.to_numpy(dtype=float))
        value_norm = value / max(float(value.max()), 1e-9)
        final = 0.65 * relevance + 0.2 * conversion + 0.15 * value_norm
        policy_epsilon = self.epsilon if assignment == "treatment" else 0.0
        indices, selection_probability = EpsilonGreedyPolicy(policy_epsilon).order(final, request.k)
        recommendations = []
        for rank, (idx, policy_score) in enumerate(
            zip(indices, selection_probability, strict=True), 1
        ):
            row = frame.iloc[int(idx)]
            recommendations.append(
                Recommendation(
                    item_id=str(row.item_id),
                    score=float(final[idx]),
                    rank=rank,
                    component_scores=ComponentScores(
                        retrieval=float(retrieval[str(row.item_id)]),
                        relevance=float(relevance[idx]),
                        propensity=float(conversion[idx]),
                        value=float(value_norm[idx]),
                        policy=float(policy_score),
                    ),
                    reason=(
                        f"{sources[str(row.item_id)]}; category_match={bool(row.category_match)}"
                    ),
                )
            )
        return RecommendationResponse(
            model_version=self.ranker.version,
            experiment_assignment=assignment,
            timestamp=datetime.now(UTC),
            recommendations=recommendations,
            metadata={
                "candidate_count": len(frame),
                "policy": "epsilon_greedy",
                "epsilon": policy_epsilon,
            },
        )
