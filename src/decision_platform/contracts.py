from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

EventType = Literal["impression", "click", "save", "conversion"]


class Context(BaseModel):
    device: Literal["mobile", "desktop", "tablet"] = "mobile"
    location_category: Literal["urban", "suburban", "rural"] = "urban"
    hour: int = Field(default=12, ge=0, le=23)
    session_depth: int = Field(default=1, ge=1, le=100)


class RecommendationRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    context: Context = Field(default_factory=Context)
    k: int = Field(default=10, ge=1, le=50)
    experiment_id: str | None = Field(default="ranking-policy-v1", max_length=128)


class ComponentScores(BaseModel):
    retrieval: float
    relevance: float
    propensity: float
    value: float
    policy: float


class Recommendation(BaseModel):
    item_id: str
    score: float
    rank: int = Field(ge=1)
    component_scores: ComponentScores
    reason: str


class RecommendationResponse(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    model_version: str
    experiment_assignment: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    recommendations: list[Recommendation]
    metadata: dict[str, Any] = Field(default_factory=dict)


class InteractionEvent(BaseModel):
    event_id: UUID = Field(default_factory=uuid4)
    user_id: str
    item_id: str
    event_type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session_id: str
    experiment_id: str | None = None
    assignment: str | None = None
    request_id: UUID | None = None
    rank: int | None = None
    exposure_propensity: float | None = Field(default=None, gt=0, le=1)
    schema_version: int = 1

    @field_validator("timestamp")
    @classmethod
    def timestamp_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        return value
