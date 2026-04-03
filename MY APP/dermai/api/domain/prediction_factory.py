
from typing import Optional

import numpy as np

from api.config import get_settings
from api.domain.disease_config import DiseaseConfig, DiseaseInfo
from api.schemas.prediction import PredictionResponse

settings = get_settings()

DISCLAIMER = (
    "This tool is for informational and educational purposes only. "
    "It is NOT a medical diagnosis. Consult a qualified dermatologist "
    "for any medical concerns."
)


class PredictionResponseFactory:
    

    def __init__(self, disease_config: DiseaseConfig) -> None:
        self._disease_config = disease_config

    def build(
        self,
        probabilities: np.ndarray,           
        gradcam_base64: Optional[str],
        gradcam_error: Optional[str] = None,
    ) -> PredictionResponse:
        
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])

        disease: Optional[DiseaseInfo] = self._disease_config.get_by_index(class_index)

      
        if disease is None:
            disease_name = f"Unknown class {class_index}"
            description = "No description available for this class."
            learn_more_url = "https://www.aad.org"
        else:
            disease_name = disease.name
            description = disease.description
            learn_more_url = disease.learn_more_url

        
        very_low = confidence < settings.very_low_confidence_threshold
        low = confidence < settings.low_confidence_threshold
        confidence_level = "very_low" if very_low else ("low" if low else "high")
        low_confidence_warning = low or very_low

        return PredictionResponse(
            disease_name=disease_name,
            confidence_score=round(confidence, 4),
            confidence_level=confidence_level,
            low_confidence_warning=low_confidence_warning,
            clinical_description=description,
            learn_more_url=learn_more_url,
            disclaimer=DISCLAIMER,
            gradcam_image_base64=gradcam_base64,
            gradcam_error=gradcam_error,
        )
