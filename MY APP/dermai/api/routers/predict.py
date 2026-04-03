
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status

from api.config import get_settings
from api.domain.disease_config import DiseaseConfig
from api.domain.model_manager import ModelManager, ModelNotLoadedError
from api.domain.prediction_factory import PredictionResponseFactory
from api.events.prediction_events import PredictionEvent, PredictionEventPublisher, LoggingListener
from api.routers.auth import get_current_user
from api.schemas.auth import UserResponse
from api.schemas.prediction import DiseaseInfo, HealthResponse, PredictionResponse
from api.services.gradcam import GradCAMService
from api.services.image_validation import ImageValidationError, ImageValidationService
from api.services.skin_detector import SkinDetectionService
from api.strategies.preprocessing import MobileNetV2PreprocessingStrategy

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Prediction"])
settings = get_settings()

_disease_config = DiseaseConfig()
_preprocessor = MobileNetV2PreprocessingStrategy()
_validator = ImageValidationService()
_skin_detector = SkinDetectionService()
_gradcam_service = GradCAMService(target_layer_name=settings.gradcam_target_layer)
_factory = PredictionResponseFactory(_disease_config)
_event_publisher = PredictionEventPublisher()
_event_publisher.register(LoggingListener())




@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: Annotated[UploadFile, File(description="Skin image (JPEG/PNG/BMP, max 10 MB)")],
    current_user: UserResponse = Depends(get_current_user),
):
    
    file_bytes = await file.read()

    
    try:
        pil_image = _validator.validate_and_open(file_bytes, file.filename or "")
    except ImageValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))

    
    skin_result = _skin_detector.check(pil_image)
    if not skin_result.is_skin:
        raise HTTPException(
            status_code=422,
            detail=skin_result.message,
        )

    
    inference_tensor, display_image = _preprocessor.preprocess(pil_image)

    
    manager = ModelManager.get_instance()
    try:
        probabilities = manager.predict(inference_tensor)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        logger.error("Inference error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Model inference failed.")

    
    import numpy as np
    class_index = int(np.argmax(probabilities))

    gradcam_b64 = None
    gradcam_error = None
    try:
        gradcam_b64 = _gradcam_service.generate(
            model=manager.model,
            inference_tensor=inference_tensor,
            display_image=display_image,
            class_index=class_index,
        )
        if gradcam_b64 is None:
            gradcam_error = "Heatmap could not be generated for this image."
    except Exception as exc:
        logger.error("Grad-CAM error: %s", exc)
        gradcam_error = "Heatmap generation encountered an error."

    
    response = _factory.build(probabilities, gradcam_b64, gradcam_error)

    
    _event_publisher.notify(PredictionEvent(
        disease_name=response.disease_name,
        confidence_score=response.confidence_score,
        confidence_level=response.confidence_level,
        gradcam_generated=gradcam_b64 is not None,
    ))

    del inference_tensor, display_image, probabilities, file_bytes

    return response


@router.get("/health", response_model=HealthResponse)
def health():
    
    manager = ModelManager.get_instance()
    if not manager.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )
    return HealthResponse(status="ok", model_loaded=True)


@router.get("/diseases", response_model=list[DiseaseInfo])
def list_diseases():
  
    return [
        DiseaseInfo(
            class_id=d.class_id,
            name=d.name,
            icd10=d.icd10,
            learn_more_url=d.learn_more_url,
            description=d.description,
        )
        for d in _disease_config.all_diseases()
    ]