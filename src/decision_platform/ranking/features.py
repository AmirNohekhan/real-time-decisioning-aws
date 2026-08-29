import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "tenure_days",
    "engagement",
    "price_sensitivity",
    "activity_frequency",
    "price",
    "quality",
    "popularity",
    "freshness_days",
    "category_match",
    "price_fit",
    "hour",
    "session_depth",
    "device_mobile",
    "device_desktop",
    "location_urban",
    "retrieval_score",
]


def build_training_frame(
    interactions: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame
) -> pd.DataFrame:
    frame = interactions.merge(users, on="user_id", validate="many_to_one").merge(
        items, on="item_id", validate="many_to_one"
    )
    frame["category_match"] = (frame.preferred_category == frame.category).astype(float)
    price_position = np.clip(np.log1p(frame.price.astype(float)) / 8, 0, 1)
    frame["price_fit"] = -(frame.price_sensitivity - price_position).abs()
    frame["device_mobile"] = (frame.device == "mobile").astype(float)
    frame["device_desktop"] = (frame.device == "desktop").astype(float)
    frame["location_urban"] = (frame.location_category == "urban").astype(float)
    # Historical item popularity proxy, intentionally not a target leak.
    frame["retrieval_score"] = frame.popularity / max(float(frame.popularity.max()), 1e-9)
    return frame


def build_inference_frame(
    user: dict[str, object],
    items: pd.DataFrame,
    context: dict[str, object],
    retrieval_scores: dict[str, float],
) -> pd.DataFrame:
    frame = items.copy()
    for key, value in user.items():
        if key != "user_id":
            frame[key] = value
    frame["category_match"] = (frame.preferred_category == frame.category).astype(float)
    frame["price_fit"] = -(
        frame.price_sensitivity.astype(float)
        - np.clip(np.log1p(frame.price.astype(float)) / 8, 0, 1)
    ).abs()
    frame["hour"] = context["hour"]
    frame["session_depth"] = context["session_depth"]
    frame["device_mobile"] = float(context["device"] == "mobile")
    frame["device_desktop"] = float(context["device"] == "desktop")
    frame["location_urban"] = float(context["location_category"] == "urban")
    frame["retrieval_score"] = frame.item_id.map(retrieval_scores).fillna(0)
    return frame
