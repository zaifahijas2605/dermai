
import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


RAW_PIXEL_THRESHOLD   = 0.25  

COHERENT_THRESHOLD    = 0.20   


@dataclass
class SkinDetectionResult:
    is_skin: bool
    skin_fraction: float
    message: str


class SkinDetectionService:

    def check(self, pil_image: Image.Image) -> SkinDetectionResult:
        try:
            img_rgb = np.array(pil_image.convert("RGB"), dtype=np.uint8)

           
            small = cv2.resize(img_rgb, (128, 128))
            img_bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)

           
            img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
            y_ch  = img_ycrcb[:, :, 0]
            cr_ch = img_ycrcb[:, :, 1]
            cb_ch = img_ycrcb[:, :, 2]

            skin_mask = (
                (y_ch  >= 50)  & (y_ch  <= 230) &
                (cr_ch >= 138) & (cr_ch <= 175) &   
                (cb_ch >= 90)  & (cb_ch <= 128)    
            ).astype(np.uint8) * 255

            total_pixels  = 128 * 128
            raw_skin      = int(np.sum(skin_mask > 0))
            raw_fraction  = raw_skin / total_pixels

            logger.info("Skin raw pixels: %.1f%%", raw_fraction * 100)

            if raw_fraction < RAW_PIXEL_THRESHOLD:
                return self._reject(raw_fraction)

           
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            closed = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
           
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

           
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                opened, connectivity=8
            )

            if num_labels < 2:
                
                return self._reject(raw_fraction)

            
            largest_area = max(stats[1:, cv2.CC_STAT_AREA]) if num_labels > 1 else 0
            coherent_fraction = largest_area / total_pixels

            logger.info(
                "Skin coherent region: %.1f%% (threshold: %.1f%%)",
                coherent_fraction * 100,
                COHERENT_THRESHOLD * 100,
            )

            if coherent_fraction >= COHERENT_THRESHOLD:
                return SkinDetectionResult(
                    is_skin=True,
                    skin_fraction=coherent_fraction,
                    message="Skin detected.",
                )
            else:
                return self._reject(coherent_fraction)

        except Exception as exc:
           
            logger.warning("Skin detection failed (failing open): %s", exc)
            return SkinDetectionResult(
                is_skin=True,
                skin_fraction=0.0,
                message="Skin detection skipped due to error.",
            )

    def _reject(self, fraction: float) -> SkinDetectionResult:
        return SkinDetectionResult(
            is_skin=False,
            skin_fraction=fraction,
            message=(
                f"No skin lesion detected in this image "
                f"({fraction*100:.1f}% skin-tone area found). "
                f"Please upload a clear, close-up photograph of a skin lesion."
            ),
        )