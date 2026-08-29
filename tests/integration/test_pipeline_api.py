from fastapi.testclient import TestClient

from decision_platform.config import get_settings
from decision_platform.data.synthetic import generate_dataset
from decision_platform.serving.api import app
from decision_platform.serving.dependencies import get_engine
from decision_platform.training.pipeline import train


def test_end_to_end_training_and_api(tmp_path, monkeypatch) -> None:
    data_dir, artifact_dir = tmp_path / "data", tmp_path / "artifacts"
    generate_dataset(60, 40, 1000, seed=9).save(data_dir)
    metrics = train(data_dir, artifact_dir, seed=9)
    assert 0 <= metrics["ndcg_at_k"] <= 1
    monkeypatch.setenv("DP_DATA_DIR", str(data_dir))
    monkeypatch.setenv("DP_ARTIFACT_DIR", str(artifact_dir))
    get_settings.cache_clear()
    get_engine.cache_clear()
    with TestClient(app) as client:
        response = client.post(
            "/v1/recommendations",
            json={
                "user_id": "u00001",
                "k": 5,
                "context": {
                    "device": "mobile",
                    "hour": 18,
                    "location_category": "urban",
                    "session_depth": 2,
                },
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert len(body["recommendations"]) == 5
    assert [x["rank"] for x in body["recommendations"]] == [1, 2, 3, 4, 5]
    assert all("component_scores" in x for x in body["recommendations"])
