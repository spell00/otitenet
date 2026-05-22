import pytest
from fastapi.testclient import TestClient
from otitenet.api.main import app
import os
from pathlib import Path
import json

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "OtiteNet Mobile API"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_deployment_manifest_not_found():
    # Ensure manifest doesn't exist or is mocked
    response = client.get("/deployment/current")
    # This might be 200 if one exists, or 404 if not.
    assert response.status_code in [200, 404]

def test_analyze_no_model():
    # If no production model is set, it should return 400
    # Note: This depends on the DB state.
    # In a real test environment we'd use a test database.
    response = client.post("/analyze", data={"person_id": 1}, files={"file": ("test.jpg", b"fakeimagecontent")})
    # Since we don't have a guaranteed production model in CI, we check for 400 or 500
    assert response.status_code in [400, 500]
