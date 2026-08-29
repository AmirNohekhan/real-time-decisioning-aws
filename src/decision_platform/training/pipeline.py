import json
from pathlib import Path
from typing import Any

from decision_platform.data.synthetic import load_dataset
from decision_platform.ranking.features import build_training_frame
from decision_platform.ranking.metrics import ranking_metrics
from decision_platform.ranking.model import RankingModel
from decision_platform.retrieval.strategies import (
    CollaborativeRetriever,
    PopularityRetriever,
    recall_at_k,
)


def temporal_split(frame: Any, fraction: float = 0.8) -> tuple[Any, Any]:
    ordered = frame.sort_values("timestamp")
    boundary = int(len(ordered) * fraction)
    return ordered.iloc[:boundary].copy(), ordered.iloc[boundary:].copy()


def train(data_dir: Path, artifact_dir: Path, seed: int = 42) -> dict[str, float]:
    dataset = load_dataset(data_dir)
    train_events, test_events = temporal_split(dataset.interactions)
    train_frame = build_training_frame(train_events, dataset.users, dataset.items)
    test_frame = build_training_frame(test_events, dataset.users, dataset.items)
    ranker = RankingModel.create(seed=seed, version="local-v1").fit(train_frame)
    metrics = ranking_metrics(test_frame, ranker.predict(test_frame), k=10)
    retrievers: tuple[PopularityRetriever | CollaborativeRetriever, ...] = (
        PopularityRetriever(),
        CollaborativeRetriever(seed=seed),
    )
    for retriever in retrievers:
        retriever.fit(train_events, dataset.items)
        metrics[f"{retriever.name}_recall_at_50"] = recall_at_k(retriever, test_events, 50)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ranker.save(artifact_dir / "ranker.joblib")
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metrics


def evaluate(artifact_dir: Path) -> dict[str, float]:
    path = artifact_dir / "metrics.json"
    if not path.exists():
        raise FileNotFoundError("metrics missing; run training first")
    return {str(k): float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
