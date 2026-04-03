
import io
import numpy as np
import pytest
from PIL import Image


from api.services.skin_detector import SkinDetectionService, RAW_PIXEL_THRESHOLD as SKIN_PIXEL_THRESHOLD

detector = SkinDetectionService()


def make_skin_image(width=200, height=200) -> Image.Image:
   
    return Image.new("RGB", (width, height), color=(200, 150, 120))


def make_green_image(width=200, height=200) -> Image.Image:
    
    return Image.new("RGB", (width, height), color=(34, 139, 34))


def make_blue_image(width=200, height=200) -> Image.Image:
    
    return Image.new("RGB", (width, height), color=(30, 100, 200))


def make_black_image(width=200, height=200) -> Image.Image:
    return Image.new("RGB", (width, height), color=(0, 0, 0))


def make_white_image(width=200, height=200) -> Image.Image:
    return Image.new("RGB", (width, height), color=(255, 255, 255))


def make_dark_skin_image(width=200, height=200) -> Image.Image:
   
    return Image.new("RGB", (width, height), color=(141, 85, 36))


def make_light_skin_image(width=200, height=200) -> Image.Image:
    
    return Image.new("RGB", (width, height), color=(255, 220, 185))




def test_typical_skin_tone_detected():
    result = detector.check(make_skin_image())
    assert result.is_skin is True
    assert result.skin_fraction > SKIN_PIXEL_THRESHOLD


def test_dark_skin_detected():
    result = detector.check(make_dark_skin_image())
    assert result.is_skin is True


def test_light_skin_detected():
    result = detector.check(make_light_skin_image())
    assert result.is_skin is True




def test_green_image_rejected():
    result = detector.check(make_green_image())
    assert result.is_skin is False


def test_blue_image_rejected():
    result = detector.check(make_blue_image())
    assert result.is_skin is False


def test_black_image_rejected():
    result = detector.check(make_black_image())
    assert result.is_skin is False



def test_result_has_message():
    result = detector.check(make_green_image())
    assert isinstance(result.message, str)
    assert len(result.message) > 0


def test_rejection_message_contains_percentage():
    result = detector.check(make_green_image())
    assert "%" in result.message


def test_skin_fraction_between_zero_and_one():
    for img in [make_skin_image(), make_green_image(), make_blue_image()]:
        result = detector.check(img)
        assert 0.0 <= result.skin_fraction <= 1.0


def test_accepts_rgba_input():
    
    img = Image.new("RGBA", (200, 200), color=(200, 150, 120, 255))
    result = detector.check(img)
    assert result.is_skin is True


def test_fails_open_on_corrupt_input():
    
    from unittest.mock import patch, MagicMock
    bad_image = MagicMock()
    bad_image.convert.side_effect = Exception("corrupt")
    result = detector.check(bad_image)
    assert result.is_skin is True   