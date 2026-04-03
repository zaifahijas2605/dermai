
import io
import uuid

import numpy as np
import pytest
from PIL import Image

from api.strategies.preprocessing import MobileNetV2PreprocessingStrategy
from api.services.jwt_service import JWTService



preprocessor = MobileNetV2PreprocessingStrategy()


def make_pil_image(width=224, height=224, color=(128, 64, 200)) -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


def test_output_shapes():
    pil = make_pil_image()
    tensor, display = preprocessor.preprocess(pil)
    assert tensor.shape == (1, 224, 224, 3)
    assert display.shape == (224, 224, 3)


def test_inference_tensor_range():
    pil = make_pil_image()
    tensor, _ = preprocessor.preprocess(pil)
    assert tensor.min() >= -1.0 - 1e-5
    assert tensor.max() <= 1.0 + 1e-5


def test_black_pixel_maps_to_minus_one():
    pil = Image.new("RGB", (224, 224), color=(0, 0, 0))
    tensor, _ = preprocessor.preprocess(pil)
    assert abs(tensor.min() - (-1.0)) < 0.01


def test_white_pixel_maps_to_plus_one():
    pil = Image.new("RGB", (224, 224), color=(255, 255, 255))
    tensor, _ = preprocessor.preprocess(pil)
    assert abs(tensor.max() - 1.0) < 0.01


def test_display_image_is_uint8():
    pil = make_pil_image()
    _, display = preprocessor.preprocess(pil)
    assert display.dtype == np.uint8


def test_display_image_range():
    pil = make_pil_image()
    _, display = preprocessor.preprocess(pil)
    assert display.min() >= 0
    assert display.max() <= 255


def test_resize_to_224():
    pil = make_pil_image(width=512, height=300)
    tensor, display = preprocessor.preprocess(pil)
    assert tensor.shape == (1, 224, 224, 3)
    assert display.shape == (224, 224, 3)


def test_inference_tensor_dtype():
    pil = make_pil_image()
    tensor, _ = preprocessor.preprocess(pil)
    assert tensor.dtype == np.float32


def test_display_and_tensor_are_independent():
    
    pil = make_pil_image()
    tensor, display = preprocessor.preprocess(pil)
    original_tensor_val = tensor[0, 0, 0, 0]
    display[0, 0, 0] = 0
    assert tensor[0, 0, 0, 0] == original_tensor_val




jwt = JWTService()
TEST_USER_ID = uuid.uuid4()
TEST_EMAIL = "jwt_test@example.com"


def test_create_access_token():
    token = jwt.create_access_token(TEST_USER_ID, TEST_EMAIL)
    assert isinstance(token, str)
    assert len(token) > 20


def test_verify_access_token():
    token = jwt.create_access_token(TEST_USER_ID, TEST_EMAIL)
    data = jwt.verify_access_token(token)
    assert data["sub"] == str(TEST_USER_ID)


def test_create_refresh_token():
    token = jwt.create_refresh_token(TEST_USER_ID)
    assert isinstance(token, str)


def test_verify_refresh_token():
    token = jwt.create_refresh_token(TEST_USER_ID)
    data = jwt.verify_refresh_token(token)
    assert data["sub"] == str(TEST_USER_ID)


def test_tampered_token_rejected():
    token = jwt.create_access_token(TEST_USER_ID, TEST_EMAIL)
    tampered = token[:-5] + "XXXXX"
    from jose import JWTError
    with pytest.raises(JWTError):
        jwt.verify_access_token(tampered)


def test_refresh_token_rejected_as_access():
    refresh = jwt.create_refresh_token(TEST_USER_ID)
    from jose import JWTError
    with pytest.raises(JWTError):
        jwt.verify_access_token(refresh)


def test_access_token_rejected_as_refresh():
    access = jwt.create_access_token(TEST_USER_ID, TEST_EMAIL)
    from jose import JWTError
    with pytest.raises(JWTError):
        jwt.verify_refresh_token(access)


def test_revoked_refresh_token_rejected():
    token = jwt.create_refresh_token(TEST_USER_ID)
    jwt.revoke_refresh_token(token)
    from jose import JWTError
    with pytest.raises(JWTError):
        jwt.verify_refresh_token(token)


def test_token_pair_generation():
    fresh_id = uuid.uuid4()   
    access, refresh = jwt.create_token_pair(fresh_id, "pair_test@example.com")
    assert access != refresh
    data_a = jwt.verify_access_token(access)
    data_r = jwt.verify_refresh_token(refresh)
    assert data_a["sub"] == str(fresh_id)
    assert data_r["sub"] == str(fresh_id)