from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Candidate:
    item_id: str
    score: float
    source: str


class CandidateGenerator(Protocol):
    name: str

    def fit(self, interactions: object, items: object) -> "CandidateGenerator": ...
    def retrieve(self, user_id: str, k: int) -> list[Candidate]: ...


class RetrievalService(Protocol):
    def retrieve(self, user_id: str, k: int) -> list[Candidate]: ...
