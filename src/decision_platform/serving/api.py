from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status

from decision_platform.config import get_settings
from decision_platform.contracts import (
    InteractionEvent,
    RecommendationRequest,
    RecommendationResponse,
)
from decision_platform.data.events import LocalEventStore
from decision_platform.serving.dependencies import get_engine
from decision_platform.serving.engine import DecisionEngine

app = FastAPI(title="Real-Time Decisioning API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/v1/recommendations", response_model=RecommendationResponse)
def recommend(
    request: RecommendationRequest, engine: Annotated[DecisionEngine, Depends(get_engine)]
) -> RecommendationResponse:
    try:
        return engine.recommend(request)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model artifacts are unavailable; run make demo",
        ) from exc


@app.post("/v1/events", status_code=status.HTTP_202_ACCEPTED)
def event(event: InteractionEvent) -> dict[str, bool]:
    store = LocalEventStore(get_settings().data_dir / "outcomes.jsonl")
    return {"accepted": store.put(event)}
