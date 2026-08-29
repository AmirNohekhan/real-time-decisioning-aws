from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticDataset:
    users: pd.DataFrame
    items: pd.DataFrame
    interactions: pd.DataFrame

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.users.to_csv(directory / "users.csv", index=False)
        self.items.to_csv(directory / "items.csv", index=False)
        self.interactions.to_csv(directory / "interactions.csv", index=False)


def generate_dataset(
    n_users: int = 500, n_items: int = 300, n_interactions: int = 20_000, seed: int = 42
) -> SyntheticDataset:
    """Generate non-trivial implicit feedback from hidden user/item factors."""
    rng = np.random.default_rng(seed)
    categories = np.array(["a", "b", "c", "d", "e", "f"])
    user_latent = rng.normal(size=(n_users, 8))
    item_latent = rng.normal(size=(n_items, 8))
    users = pd.DataFrame(
        {
            "user_id": [f"u{i:05d}" for i in range(n_users)],
            "tenure_days": rng.integers(1, 2500, n_users),
            "engagement": rng.beta(2, 5, n_users),
            "price_sensitivity": rng.beta(2, 2, n_users),
            "activity_frequency": rng.lognormal(1, 0.6, n_users),
            "preferred_category": rng.choice(categories, n_users),
        }
    )
    price = rng.lognormal(3.5, 0.8, n_items)
    items = pd.DataFrame(
        {
            "item_id": [f"i{i:05d}" for i in range(n_items)],
            "category": rng.choice(categories, n_items),
            "price": price,
            "quality": rng.beta(5, 2, n_items),
            "popularity": rng.pareto(2.5, n_items) + 0.05,
            "freshness_days": rng.exponential(120, n_items).astype(int),
            "available": rng.random(n_items) > 0.04,
        }
    )
    user_idx = rng.integers(0, n_users, n_interactions)
    # Mix personalized latent retrieval with popularity exposure to induce realistic bias.
    random_items = rng.integers(0, n_items, n_interactions)
    popular_items = rng.choice(n_items, n_interactions, p=items.popularity / items.popularity.sum())
    item_idx = np.where(rng.random(n_interactions) < 0.65, popular_items, random_items)
    device = rng.choice(["mobile", "desktop", "tablet"], n_interactions, p=[0.62, 0.31, 0.07])
    hour = rng.integers(0, 24, n_interactions)
    latent_affinity = np.sum(user_latent[user_idx] * item_latent[item_idx], axis=1) / 3
    category_match = (
        users.preferred_category.to_numpy()[user_idx] == items.category.to_numpy()[item_idx]
    ).astype(float)
    price_fit = -np.abs(
        users.price_sensitivity.to_numpy()[user_idx] - np.clip(np.log1p(price[item_idx]) / 8, 0, 1)
    )
    context_effect = 0.25 * (device == "desktop") + 0.2 * ((hour >= 18) & (hour <= 22))
    logit = (
        -2.5
        + 0.75 * latent_affinity
        + 0.9 * category_match
        + 0.7 * price_fit
        + 0.5 * items.quality.to_numpy()[item_idx]
        + context_effect
    )
    click_prob = 1 / (1 + np.exp(-np.clip(logit, -15, 15)))
    clicked = rng.random(n_interactions) < click_prob
    conversion_prob = click_prob * (0.08 + 0.15 * items.quality.to_numpy()[item_idx])
    converted = clicked & (rng.random(n_interactions) < conversion_prob)
    event = np.where(converted, "conversion", np.where(clicked, "click", "impression"))
    start = pd.Timestamp("2025-01-01", tz="UTC")
    timestamps = start + pd.to_timedelta(rng.integers(0, 120 * 86400, n_interactions), unit="s")
    interactions = pd.DataFrame(
        {
            "event_id": [f"e{i:08d}" for i in range(n_interactions)],
            "user_id": users.user_id.to_numpy()[user_idx],
            "item_id": items.item_id.to_numpy()[item_idx],
            "event_type": event,
            "label": clicked.astype(int) + converted.astype(int) * 2,
            "timestamp": timestamps,
            "session_id": [f"s{i // 5:07d}" for i in range(n_interactions)],
            "device": device,
            "location_category": rng.choice(["urban", "suburban", "rural"], n_interactions),
            "hour": hour,
            "session_depth": rng.integers(1, 12, n_interactions),
            "exposure_propensity": np.clip(
                0.65 * items.popularity.to_numpy()[item_idx] / items.popularity.max() + 0.05,
                0.05,
                1,
            ),
            "schema_version": 1,
        }
    ).sort_values("timestamp", ignore_index=True)
    return SyntheticDataset(users, items, interactions)


def load_dataset(directory: Path) -> SyntheticDataset:
    return SyntheticDataset(
        pd.read_csv(directory / "users.csv"),
        pd.read_csv(directory / "items.csv"),
        pd.read_csv(directory / "interactions.csv", parse_dates=["timestamp"]),
    )
