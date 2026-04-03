
import io
import struct
import zlib

import pytest
from PIL import Image

from api.services.image_validation import ImageValidationError, ImageValidationService

validator = ImageValidationService()


def make_jpeg(width=100, height=100, mode="RGB") -> bytes:
    img = Image.new(mode, (width, height), color=(200, 150, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_png(width=100, height=100, mode="RGB") -> bytes:
    img = Image.new(mode, (width, height), color=(200, 150, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_bmp(width=100, height=100) -> bytes:
    img = Image.new("RGB", (width, height), color=(200, 150, 120))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    return buf.getvalue()




def test_valid_jpeg():
    img = validator.validate_and_open(make_jpeg(), "test.jpg")
    assert img is not None
    assert img.mode == "RGB"


def test_valid_png():
    img = validator.validate_and_open(make_png(), "test.png")
    assert img is not None
    assert img.mode == "RGB"


def test_valid_bmp():
    img = validator.validate_and_open(make_bmp(), "test.bmp")
    assert img is not None

def test_grayscale_converted_to_rgb():
    img = Image.new("L", (100, 100), color=128) 
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    result = validator.validate_and_open(buf.getvalue(), "gray.jpg")
    assert result.mode == "RGB"


def test_rgba_converted_to_rgb():
    png_img = Image.new("RGBA", (100, 100), (200, 150, 120, 255))
    buf = io.BytesIO()
    png_img.save(buf, format="PNG")
    img = validator.validate_and_open(buf.getvalue(), "rgba.png")
    assert img.mode == "RGB"




def test_empty_file_rejected():
    with pytest.raises(ImageValidationError) as exc:
        validator.validate_and_open(b"", "empty.jpg")
    assert exc.value.status_code == 413


def test_oversized_file_rejected():
    huge = b"\xff\xd8\xff" + b"x" * (11 * 1024 * 1024)
    with pytest.raises(ImageValidationError) as exc:
        validator.validate_and_open(huge, "big.jpg")
    assert exc.value.status_code == 413




def test_spoofed_extension_rejected():
    
    png_data = make_png()
    
    img = validator.validate_and_open(png_data, "fake.jpg")
    assert img is not None


def test_random_bytes_rejected():
    with pytest.raises(ImageValidationError) as exc:
        validator.validate_and_open(b"this is not an image at all!!!", "fake.jpg")
    assert exc.value.status_code == 422


def test_text_file_rejected():
    with pytest.raises(ImageValidationError) as exc:
        validator.validate_and_open(b"hello world plain text", "text.jpg")
    assert exc.value.status_code == 422



def test_too_small_rejected():
    data = make_jpeg(width=32, height=32)
    with pytest.raises(ImageValidationError) as exc:
        validator.validate_and_open(data, "tiny.jpg")
    assert exc.value.status_code == 422


def test_minimum_valid_size():
    data = make_jpeg(width=64, height=64)
    img = validator.validate_and_open(data, "min.jpg")
    assert img is not None


def test_large_valid_image():
    data = make_jpeg(width=1024, height=1024)
    img = validator.validate_and_open(data, "large.jpg")
    assert img is not None