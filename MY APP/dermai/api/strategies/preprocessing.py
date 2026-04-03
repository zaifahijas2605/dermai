
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from PIL import Image

TARGET_SIZE = (224, 224)


class BasePreprocessingStrategy(ABC):
    

    @abstractmethod
    def preprocess(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
       
        ...


class MobileNetV2PreprocessingStrategy(BasePreprocessingStrategy):
   

    def preprocess(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        
        resized = image.resize(TARGET_SIZE, Image.BILINEAR)

        
        display_image = np.array(resized, dtype=np.uint8)

        arr = np.array(resized, dtype=np.float32)

        
        arr = arr / 127.5 - 1.0

        
        inference_tensor = np.expand_dims(arr, axis=0)

        return inference_tensor, display_image
