
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.config import get_settings
from api.database import init_db
from api.domain.model_manager import ModelManager
from api.routers import auth, predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
settings = get_settings()




@asynccontextmanager
async def lifespan(app: FastAPI):
    

    
    logger.info("=== DermAI Classifier starting up ===")

    
    logger.info("Initialising database …")
    try:
        init_db()
        logger.info("Database ready.")
    except Exception as exc:
        logger.critical("Database initialisation failed: %s", exc)
       
    logger.info("Loading ML model from: %s", settings.model_path)
    manager = ModelManager.get_instance()
    try:
        manager.load(settings.model_path)
    except FileNotFoundError:
        logger.critical(
            "MODEL FILE NOT FOUND at '%s'. "
            "Place your .keras file there and restart. "
            "The /predict endpoint will return 503 until the model is loaded.",
            settings.model_path,
        )
    except RuntimeError as exc:
        logger.critical("Model loading error: %s", exc)

    logger.info("=== Startup complete. API is ready. ===")
    yield

    
    logger.info("=== DermAI Classifier shutting down ===")




def create_app() -> FastAPI:
    app = FastAPI(
        title="DermAI Classifier",
        description=(
            "Skin disease classification API using MobileNetV2 + SE attention. "
            "Provides disease name, confidence score, clinical description, "
            "Learn More URL, and Grad-CAM++ heatmap. "
            "No medical data is stored."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred. Please try again."},
        )

    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(predict.router, prefix="/api/v1")

    
    import os
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
    if os.path.isdir(frontend_path):
        app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

    return app


app = create_app()
