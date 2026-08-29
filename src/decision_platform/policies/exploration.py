from dataclasses import dataclass

import numpy as np


@dataclass
class EpsilonGreedyPolicy:
    epsilon: float = 0.05
    seed: int = 42

    def order(self, scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        rng = np.random.default_rng(self.seed)
        remaining = list(np.argsort(-scores))
        selected: list[int] = []
        propensities: list[float] = []
        while remaining and len(selected) < k:
            explore = rng.random() < self.epsilon
            pick = int(rng.integers(len(remaining))) if explore else 0
            selected.append(remaining.pop(pick))
            n = len(remaining) + 1
            propensities.append(self.epsilon / n + (1 - self.epsilon if pick == 0 else 0))
        return np.asarray(selected), np.asarray(propensities)


def simulate_policy(rounds: int = 2000, epsilon: float = 0.1, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    true_rates = np.array([0.08, 0.10, 0.13, 0.09, 0.11])
    estimates = np.zeros(5)
    counts = np.zeros(5)
    reward = 0.0
    for _ in range(rounds):
        arm = int(rng.integers(5)) if rng.random() < epsilon else int(np.argmax(estimates))
        observed = float(rng.random() < true_rates[arm])
        counts[arm] += 1
        estimates[arm] += (observed - estimates[arm]) / counts[arm]
        reward += observed
    oracle = rounds * float(true_rates.max())
    return {
        "reward": reward,
        "oracle_expected_reward": oracle,
        "opportunity_cost": oracle - reward,
        "best_arm_found": float(np.argmax(estimates) == np.argmax(true_rates)),
        "min_arm_exposures": float(counts.min()),
    }
