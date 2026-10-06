from fastapi.testclient import TestClient

from ml_pipeline.api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models_endpoint():
    response = client.get("/models")
    assert response.status_code == 200
    assert "models" in response.json()
