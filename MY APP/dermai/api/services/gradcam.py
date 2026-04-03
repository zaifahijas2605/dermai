
import base64
import io
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class GradCAMService:
   

    OVERLAY_ALPHA = 0.4     

    def __init__(self, target_layer_name: str = "multiply") -> None:
        self._target_layer_name = target_layer_name

    def generate(
        self,
        model,
        inference_tensor: np.ndarray,   
        display_image: np.ndarray,       
        class_index: int,
    ) -> Optional[str]:
       
        try:
            import tensorflow as tf
            import cv2

            
            target_layer = model.get_layer(self._target_layer_name)
            grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[target_layer.output, model.output],
            )

            
            inputs_tensor = tf.cast(inference_tensor, tf.float32)

            
            with tf.GradientTape() as tape:
                outputs = grad_model(inputs_tensor, training=False)
                conv_outputs = outputs[0]

                
                pred_tensor = outputs[1]
                if isinstance(pred_tensor, (list, tuple)):
                    pred_tensor = tf.stack(pred_tensor, axis=0)
                pred_tensor = tf.reshape(pred_tensor, (1, -1)) 

                class_score = pred_tensor[0, class_index]

            
            grads = tape.gradient(class_score, conv_outputs)

            if grads is None or tf.reduce_sum(tf.abs(grads)) == 0:
                logger.warning("Grad-CAM: zero gradients for class %d — skipping heatmap.", class_index)
                return None

            
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            
            if len(conv_outputs.shape) == 4:
                conv_outputs = conv_outputs[0]

            
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  
            heatmap = tf.squeeze(heatmap)                            

           
            heatmap = heatmap.numpy()
            heatmap = np.maximum(heatmap, 0)
            heatmap_max = heatmap.max()
            if heatmap_max == 0:
                logger.warning("Grad-CAM: heatmap is all zeros — skipping.")
                return None
            heatmap /= heatmap_max

            
            heatmap_resized = cv2.resize(heatmap, (224, 224))

            
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            
            overlay = (
                display_image.astype(np.float32) * (1 - self.OVERLAY_ALPHA)
                + heatmap_colored.astype(np.float32) * self.OVERLAY_ALPHA
            ).astype(np.uint8)

            
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(overlay)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("utf-8")
            return f"data:image/png;base64,{b64}"

        except Exception as exc:
            logger.error("Grad-CAM generation failed: %s", exc, exc_info=True)
            return None