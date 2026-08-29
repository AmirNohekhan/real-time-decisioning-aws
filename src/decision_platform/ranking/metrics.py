import numpy as np
import pandas as pd


def ranking_metrics(frame: pd.DataFrame, scores: np.ndarray, k: int = 10) -> dict[str, float]:
    work = frame[["user_id", "label"]].copy()
    work["score"] = scores
    ndcg, ap, rr, precision, recall = [], [], [], [], []
    for _, group in work.groupby("user_id"):
        ranked = group.sort_values("score", ascending=False).head(k)
        rel = (ranked.label.to_numpy() > 0).astype(float)
        graded = ranked.label.to_numpy(dtype=float)
        discounts = 1 / np.log2(np.arange(2, len(ranked) + 2))
        dcg = float(np.sum((2**graded - 1) * discounts))
        ideal = np.sort(group.label.to_numpy(dtype=float))[::-1][:k]
        idcg = float(np.sum((2**ideal - 1) * discounts[: len(ideal)]))
        ndcg.append(dcg / idcg if idcg else 0.0)
        hits = np.cumsum(rel)
        positives = int((group.label > 0).sum())
        ap.append(
            float(np.sum((hits / np.arange(1, len(rel) + 1)) * rel) / max(min(positives, k), 1))
        )
        positions = np.flatnonzero(rel)
        rr.append(1 / (int(positions[0]) + 1) if len(positions) else 0.0)
        precision.append(float(rel.mean()) if len(rel) else 0.0)
        recall.append(float(rel.sum() / max(positives, 1)))
    return {
        "ndcg_at_k": float(np.mean(ndcg)),
        "map_at_k": float(np.mean(ap)),
        "mrr": float(np.mean(rr)),
        "precision_at_k": float(np.mean(precision)),
        "recall_at_k": float(np.mean(recall)),
    }
