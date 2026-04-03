
import threading
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ModelNotLoadedError(Exception):
    
 pass 

class ModelManager:
    

    _instance: Optional["ModelManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None
        self._is_loaded: bool = False
        self._model_path: Optional[str] = None

    

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    

    def load(self, path: str) -> None:
       
        import tensorflow as tf  

        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path.resolve()}")

        logger.info("Loading model from %s ...", model_path)
        try:
            self._model = tf.keras.models.load_model(str(model_path), compile=False)
            self._is_loaded = True
            self._model_path = str(model_path)
            logger.info(
                "Model loaded successfully. Input shape: %s  Output shape: %s",
                self._model.input_shape,
                self._model.output_shape,
            )
        except Exception as exc:
            self._is_loaded = False
            raise RuntimeError(f"Failed to load model: {exc}") from exc

    def is_loaded(self) -> bool:
        return self._is_loaded

    

    def predict(self, inference_tensor: np.ndarray) -> np.ndarray:
       
        if not self._is_loaded or self._model is None:
            raise ModelNotLoadedError(
                "Model is not loaded. The service is starting up or encountered an error."
            )
        probs = self._model.predict(inference_tensor, verbose=0)
        return probs[0]  

    @property
    def model(self):
       
        if not self._is_loaded:
            raise ModelNotLoadedError("Model not loaded.")
        return self._model
