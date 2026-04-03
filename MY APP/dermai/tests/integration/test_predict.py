
import io
import numpy as np
import pytest
from PIL import Image
from unittest.mock import patch, MagicMock

from api.services.skin_detector import SkinDetectionResult


def make_jpeg_bytes(width=100, height=100) -> bytes:
    img = Image.new("RGB", (width, height), color=(200, 150, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_png_bytes(width=100, height=100) -> bytes:
    img = Image.new("RGB", (width, height), color=(200, 150, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()




def test_health_model_loaded(client, mock_model):
    mock_model.is_loaded.return_value = True
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_health_model_not_loaded(client, mock_model):
    mock_model.is_loaded.return_value = False
    resp = client.get("/api/v1/health")
    assert resp.status_code == 503
    mock_model.is_loaded.return_value = True   

def test_diseases_returns_seven(client):
    resp = client.get("/api/v1/diseases")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 7


def test_diseases_have_required_fields(client):
    resp = client.get("/api/v1/diseases")
    for d in resp.json():
        assert "name" in d
        assert "class_id" in d
        assert "icd10" in d
        assert "learn_more_url" in d




def test_predict_without_auth(client, sample_jpeg_bytes):
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("skin.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 403


def test_predict_invalid_token(client, sample_jpeg_bytes):
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("skin.jpg", sample_jpeg_bytes, "image/jpeg")},
        headers={"Authorization": "Bearer invalid.token"},
    )
    assert resp.status_code == 401




def test_predict_valid_jpeg(client, auth_headers, sample_jpeg_bytes):
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("skin.jpg", sample_jpeg_bytes, "image/jpeg")},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "disease_name" in data
    assert "confidence_score" in data
    assert "confidence_level" in data
    assert "clinical_description" in data
    assert "learn_more_url" in data
    assert "disclaimer" in data


def test_predict_valid_png(client, auth_headers, sample_png_bytes):
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("skin.png", sample_png_bytes, "image/png")},
        headers=auth_headers,
    )
    assert resp.status_code == 200


def test_predict_response_confidence_in_range(client, auth_headers, sample_jpeg_bytes):
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("skin.jpg", sample_jpeg_bytes, "image/jpeg")},
        headers=auth_headers,
    )
    score = resp.json()["confidence_score"]
    assert 0.0 <= score <= 1.0


def test_predict_confidence_level_valid(client, auth_headers, sample_jpeg_bytes):
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("skin.jpg", sample_jpeg_bytes, "image/jpeg")},
        headers=auth_headers,
    )
    assert resp.json()["confidence_level"] in ("high", "low", "very_low")




def test_predict_empty_file_rejected(client, auth_headers):
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
        headers=auth_headers,
    )
    assert resp.status_code == 413


def test_predict_invalid_file_rejected(client, auth_headers):
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("fake.jpg", b"not an image", "image/jpeg")},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_predict_too_small_image_rejected(client, auth_headers):
    tiny = Image.new("RGB", (32, 32), color=(200, 150, 120))
    buf = io.BytesIO()
    tiny.save(buf, format="JPEG")
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("tiny.jpg", buf.getvalue(), "image/jpeg")},
        headers=auth_headers,
    )
    assert resp.status_code == 422




def test_predict_non_skin_image_rejected(client, auth_headers, sample_jpeg_bytes):
    
    rejection = SkinDetectionResult(
        is_skin=False,
        skin_fraction=0.02,
        message="No skin lesion detected (2.0% skin-tone pixels found).",
    )
    with patch("api.routers.predict._skin_detector") as mock_sd:
        mock_sd.check.return_value = rejection
        resp = client.post(
            "/api/v1/predict",
            files={"file": ("dog.jpg", sample_jpeg_bytes, "image/jpeg")},
            headers=auth_headers,
        )
    assert resp.status_code == 422
    assert "skin" in resp.json()["detail"].lower()


def test_predict_skin_detection_fail_open(client, auth_headers, sample_jpeg_bytes):
    
    fail_open = SkinDetectionResult(
        is_skin=True,
        skin_fraction=0.0,
        message="Skin detection skipped due to error.",
    )
    with patch("api.routers.predict._skin_detector") as mock_sd:
        mock_sd.check.return_value = fail_open
        resp = client.post(
            "/api/v1/predict",
            files={"file": ("skin.jpg", sample_jpeg_bytes, "image/jpeg")},
            headers=auth_headers,
        )
    assert resp.status_code == 200




def test_predict_model_not_loaded_returns_503(client, auth_headers, sample_jpeg_bytes, mock_model):
    from api.domain.model_manager import ModelNotLoadedError
    mock_model.predict.side_effect = ModelNotLoadedError("Model not loaded.")
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("skin.jpg", sample_jpeg_bytes, "image/jpeg")},
        headers=auth_headers,
    )
    assert resp.status_code == 503
    mock_model.predict.side_effect = None
    mock_model.predict.return_value = np.array(
        [0.05, 0.05, 0.05, 0.05, 0.70, 0.05, 0.05], dtype=np.float32
    )




def test_no_medical_tables_in_database():
    
    from sqlalchemy import inspect
    from tests.conftest import engine
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "users" in tables
    medical_tables = [t for t in tables if t != "users"]
    assert medical_tables == [], f"Unexpected tables found: {medical_tables}"

def test_predict_returns_gradcam(client, auth_headers, sample_jpeg_bytes, mock_model):
    
    import base64
    fake_b64 = "data:image/png;base64," + base64.b64encode(b"fakepngdata").decode()

    
    mock_model._mock_gc.generate.return_value = fake_b64

    resp = client.post(
        "/api/v1/predict",
        files={"file": ("skin.jpg", sample_jpeg_bytes, "image/jpeg")},
        headers=auth_headers,
    )

    
    mock_model._mock_gc.generate.return_value = None

    assert resp.status_code == 200
    data = resp.json()
    assert "gradcam_image_base64" in data
    assert data["gradcam_image_base64"] is not None
    assert len(data["gradcam_image_base64"]) > 0
    raw = data["gradcam_image_base64"].split(",")[-1]
    decoded = base64.b64decode(raw)
    assert len(decoded) > 0