
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PredictionEvent:
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    disease_name: str = ""
    confidence_score: float = 0.0
    confidence_level: str = ""
    gradcam_generated: bool = False
    error: Optional[str] = None


class PredictionListener(ABC):
   

    @abstractmethod
    def on_prediction(self, event: PredictionEvent) -> None: ...


class LoggingListener(PredictionListener):
    

    def on_prediction(self, event: PredictionEvent) -> None:
        if event.error:
            logger.error("Prediction error at %s: %s", event.timestamp, event.error)
        else:
            logger.info(
                "Prediction at %s | class=%s | confidence=%.4f | level=%s | gradcam=%s",
                event.timestamp,
                event.disease_name,
                event.confidence_score,
                event.confidence_level,
                event.gradcam_generated,
            )


class PredictionEventPublisher:
    

    def __init__(self) -> None:
        self._listeners: list[PredictionListener] = []

    def register(self, listener: PredictionListener) -> None:
        self._listeners.append(listener)

    def notify(self, event: PredictionEvent) -> None:
        for listener in self._listeners:
            try:
                listener.on_prediction(event)
            except Exception as exc:
                logger.warning("Listener %s raised an error: %s", listener, exc)
