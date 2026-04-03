
import numpy as np
import pytest

from api.domain.disease_config import DiseaseConfig
from api.domain.prediction_factory import PredictionResponseFactory


config = DiseaseConfig()
factory = PredictionResponseFactory(config)



def test_loads_seven_classes():
    assert len(config.all_diseases()) == 7


def test_class_zero_is_actinic_keratosis():
    
    d = config.get_by_index(0)
    assert d is not None
    assert "Actinic" in d.name


def test_class_one_is_basal_cell():
    d = config.get_by_index(1)
    assert "Basal" in d.name


def test_class_two_is_benign_keratosis():
    d = config.get_by_index(2)
    assert "Benign" in d.name


def test_class_three_is_dermatofibroma():
    d = config.get_by_index(3)
    assert "Dermatofibroma" in d.name


def test_class_four_is_melanoma():
    
    d = config.get_by_index(4)
    assert "Melanoma" in d.name


def test_class_five_is_melanocytic_nevi():
    
    d = config.get_by_index(5)
    assert "Nevi" in d.name or "Nevus" in d.name or "Melanocytic" in d.name


def test_class_six_is_vascular():
    d = config.get_by_index(6)
    assert "Vascular" in d.name


def test_invalid_index_returns_none():
    assert config.get_by_index(99) is None


def test_all_diseases_have_required_fields():
    for d in config.all_diseases():
        assert d.name
        assert d.icd10
        assert d.learn_more_url
        assert d.description




def make_probs(top_class: int, confidence: float) -> np.ndarray:
    
    probs = np.ones(7, dtype=np.float32) * ((1 - confidence) / 6)
    probs[top_class] = confidence
    return probs


def test_high_confidence_response():
    probs = make_probs(4, 0.82)  
    r = factory.build(probs, None, None)
    assert r.disease_name == "Melanoma"
    assert r.confidence_level == "high"
    assert r.confidence_score == pytest.approx(0.82, abs=0.01)
    assert r.low_confidence_warning is False


def test_low_confidence_response():
    probs = make_probs(5, 0.40)  
    r = factory.build(probs, None, None)
    assert r.confidence_level == "low"
    assert r.low_confidence_warning is True


def test_very_low_confidence_response():
    probs = make_probs(0, 0.20)  
    r = factory.build(probs, None, None)
    assert r.confidence_level == "very_low"
    assert r.low_confidence_warning is True


def test_gradcam_included_when_provided():
    probs = make_probs(4, 0.75)
    fake_b64 = "data:image/png;base64,abc123"
    r = factory.build(probs, fake_b64, None)
    assert r.gradcam_image_base64 == fake_b64


def test_gradcam_none_when_not_provided():
    probs = make_probs(4, 0.75)
    r = factory.build(probs, None, "Generation failed")
    assert r.gradcam_image_base64 is None
    assert r.gradcam_error == "Generation failed"


def test_response_has_all_required_fields():
    probs = make_probs(1, 0.60)
    r = factory.build(probs, None, None)
    assert r.disease_name
    assert r.confidence_score > 0
    assert r.confidence_level in ("high", "low", "very_low")
    assert r.clinical_description
    assert r.learn_more_url
    assert r.disclaimer


def test_disclaimer_is_present():
    probs = make_probs(4, 0.90)
    r = factory.build(probs, None, None)
    assert len(r.disclaimer) > 10