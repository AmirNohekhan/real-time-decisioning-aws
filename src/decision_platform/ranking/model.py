from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from decision_platform.ranking.features import FEATURE_COLUMNS


@dataclass
class RankingModel:
    """Tree ranker trained on graded relevance; ranking occurs within each request group."""

    model: HistGradientBoostingRegressor
    version: str

    @classmethod
    def create(cls, seed: int = 42, version: str = "local-v1") -> "RankingModel":
        return cls(
            HistGradientBoostingRegressor(
                max_iter=140,
                max_leaf_nodes=31,
                learning_rate=0.07,
                l2_regularization=0.1,
                random_state=seed,
            ),
            version,
        )

    def fit(self, frame: pd.DataFrame) -> "RankingModel":
        # Inverse propensity weighting partially corrects popularity-driven exposure bias.
        weights = 1 / frame.get("exposure_propensity", pd.Series(1.0, index=frame.index)).clip(
            0.05, 1
        )
        self.model.fit(frame[FEATURE_COLUMNS], frame["label"], sample_weight=weights)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(frame[FEATURE_COLUMNS]), dtype=float)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "RankingModel":
        loaded: Any = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("artifact is not a RankingModel")
        return loaded


def propensity(scores: np.ndarray) -> np.ndarray:
    return np.asarray(1 / (1 + np.exp(-np.clip(scores, -20, 20))), dtype=float)
