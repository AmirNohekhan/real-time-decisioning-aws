from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol, cast

import pandas as pd


class FeatureStore(Protocol):
    def get_user(self, user_id: str, as_of: datetime | None = None) -> Mapping[str, Any]: ...
    def get_items(self, item_ids: list[str], as_of: datetime | None = None) -> pd.DataFrame: ...


class LocalFeatureStore:
    """In-memory online-store analogue; snapshots are produced offline without future events."""

    def __init__(self, users: pd.DataFrame, items: pd.DataFrame) -> None:
        self.users = users.set_index("user_id", drop=False)
        self.items = items.set_index("item_id", drop=False)

    def get_user(self, user_id: str, as_of: datetime | None = None) -> Mapping[str, Any]:
        del as_of
        if user_id not in self.users.index:
            return {
                "user_id": user_id,
                "tenure_days": 0,
                "engagement": 0.0,
                "price_sensitivity": 0.5,
                "activity_frequency": 0.0,
                "preferred_category": "unknown",
            }
        return cast(dict[str, Any], self.users.loc[user_id].to_dict())

    def get_items(self, item_ids: list[str], as_of: datetime | None = None) -> pd.DataFrame:
        del as_of
        existing = [item_id for item_id in item_ids if item_id in self.items.index]
        return self.items.loc[existing].reset_index(drop=True)


def point_in_time_counts(interactions: pd.DataFrame) -> pd.DataFrame:
    """Historical user/item event counts strictly preceding each row's timestamp."""
    ordered = interactions.sort_values(["timestamp", "event_id"]).copy()
    ordered["user_prior_events"] = ordered.groupby("user_id").cumcount()
    ordered["item_prior_events"] = ordered.groupby("item_id").cumcount()
    return ordered
