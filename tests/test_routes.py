import pytest
from fastapi.testclient import TestClient
from app.api.routes import create_app, init_api_state
from app.services.alert_service import AlertService
from app.services.evidence_service import EvidenceService

@pytest.fixture
def client():
    app = create_app()
    
    alert_service = AlertService()
    evidence_service = EvidenceService()
    init_api_state(alert_service, evidence_service)
    
    return TestClient(app)

def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_status_endpoint(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert data["total_frames_processed"] == 0
    assert data["total_alerts"] == 0
    assert data["current_decision"] is None

def test_alerts_endpoint(client):
    response = client.get("/api/alerts")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0

def test_evidence_endpoint(client):
    response = client.get("/api/evidence")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
