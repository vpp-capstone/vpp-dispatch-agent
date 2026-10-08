from __future__ import annotations

import pytest
from api.main import app
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)
