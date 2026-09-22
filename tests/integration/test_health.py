from fastapi.testclient import TestClient

from decision_platform.config import get_settings
from decision_platform.data.synthetic import generate_dataset
from decision_platform.serving.api import app
from decision_platform.training.pipeline import train


def test_health_reports_degraded_before_artifacts_exist(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DP_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    get_settings.cache_clear()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "degraded", "dataset_ready": False, "model_ready": False}


def test_health_reports_healthy_once_dataset_and_model_exist(tmp_path, monkeypatch) -> None:
    data_dir, artifact_dir = tmp_path / "data", tmp_path / "artifacts"
    generate_dataset(20, 15, 200, seed=3).save(data_dir)
    train(data_dir, artifact_dir, seed=3)
    monkeypatch.setenv("DP_DATA_DIR", str(data_dir))
    monkeypatch.setenv("DP_ARTIFACT_DIR", str(artifact_dir))
    get_settings.cache_clear()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "healthy", "dataset_ready": True, "model_ready": True}
