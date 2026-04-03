
from typing import Optional
from pydantic import BaseModel


class ConfidenceLevel(str):
    HIGH = "high"
    LOW = "low"
    VERY_LOW = "very_low"


class PredictionResponse(BaseModel):
   
    disease_name: str
    confidence_score: float
    confidence_level: str               
    low_confidence_warning: bool
    clinical_description: str
    learn_more_url: str
    disclaimer: str
    gradcam_image_base64: Optional[str] = None   
    gradcam_error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"


class DiseaseInfo(BaseModel):
    class_id: int
    name: str
    icd10: str
    learn_more_url: str
    description: str
