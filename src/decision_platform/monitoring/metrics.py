import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def recommendation_health(recommendations: pd.DataFrame, catalog: pd.DataFrame) -> dict[str, float]:
    if recommendations.empty:
        return {"coverage": 0.0, "diversity": 0.0, "popularity_concentration": 0.0}
    coverage = recommendations.item_id.nunique() / max(catalog.item_id.nunique(), 1)
    joined = recommendations.merge(catalog[["item_id", "category", "popularity"]], on="item_id")
    pairs = joined.groupby("request_id").category.apply(lambda x: x.nunique() / max(len(x), 1))
    concentration = joined.popularity.sum() / max(float(catalog.popularity.sum()), 1e-9)
    return {
        "coverage": float(coverage),
        "diversity": float(pairs.mean()),
        "popularity_concentration": float(concentration),
    }


def population_stability(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.histogram_bin_edges(np.concatenate([reference, current]), bins=bins)
    p = np.histogram(reference, edges)[0].astype(float) + 1e-6
    q = np.histogram(current, edges)[0].astype(float) + 1e-6
    p, q = p / p.sum(), q / q.sum()
    return float(jensenshannon(p, q) ** 2)
