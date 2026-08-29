from functools import lru_cache

from decision_platform.config import get_settings
from decision_platform.data.synthetic import load_dataset
from decision_platform.features.store import LocalFeatureStore
from decision_platform.ranking.model import RankingModel
from decision_platform.retrieval.strategies import HybridRetriever
from decision_platform.serving.engine import DecisionEngine


@lru_cache
def get_engine() -> DecisionEngine:
    settings = get_settings()
    dataset = load_dataset(settings.data_dir)
    retriever = HybridRetriever().fit(dataset.interactions, dataset.items)
    ranker = RankingModel.load(settings.artifact_dir / "ranker.joblib")
    return DecisionEngine(
        LocalFeatureStore(dataset.users, dataset.items), retriever, ranker, settings.epsilon
    )
