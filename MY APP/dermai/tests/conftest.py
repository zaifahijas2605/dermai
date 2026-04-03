
import io
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import get_db
from api.main import app
from api.repositories.models import Base


TEST_DATABASE_URL = "sqlite:///./test_dermai.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="session", autouse=True)
def create_tables():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c



@pytest.fixture(scope="session", autouse=True)
def mock_model():
    mock_mgr = MagicMock()
    mock_mgr.is_loaded.return_value = True
    mock_mgr.predict.return_value = np.array(
        [0.05, 0.05, 0.05, 0.05, 0.70, 0.05, 0.05], dtype=np.float32
    )
    mock_mgr.model = MagicMock()

    with patch("api.domain.model_manager.ModelManager.get_instance", return_value=mock_mgr):
        with patch("api.routers.predict._gradcam_service") as mock_gc:
            mock_gc.generate.return_value = None
            mock_mgr._mock_gc = mock_gc  
            yield mock_mgr


@pytest.fixture(autouse=True)
def mock_skin_detector_pass():
    
    from api.services.skin_detector import SkinDetectionResult
    result = SkinDetectionResult(is_skin=True, skin_fraction=0.45, message="Skin detected.")
    with patch("api.routers.predict._skin_detector") as mock_sd:
        mock_sd.check.return_value = result
        yield mock_sd



@pytest.fixture
def registered_user(client):
    email = f"user_{uuid.uuid4().hex[:8]}@test.com"
    password = "TestPass1!"
    resp = client.post("/api/v1/auth/register", json={
        "email": email,
        "display_name": "Test User",
        "password": password,
    })
    assert resp.status_code == 201
    return {"email": email, "password": password, "display_name": "Test User"}


@pytest.fixture
def auth_headers(client, registered_user):
    resp = client.post("/api/v1/auth/login", json={
        "email": registered_user["email"],
        "password": registered_user["password"],
    })
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}



@pytest.fixture
def sample_jpeg_bytes():
    
    from PIL import Image
    img = Image.new("RGB", (100, 100), color=(200, 150, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_png_bytes():
    
    from PIL import Image
    img = Image.new("RGB", (100, 100), color=(200, 150, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()