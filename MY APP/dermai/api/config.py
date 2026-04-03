
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    
    
    database_url: str = "postgresql+psycopg://dermai_user:changeme@localhost:5432/dermai_db"
    
    jwt_secret_key: str = "change-this-secret-key-minimum-32-characters-long"
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 60
    refresh_token_ttl_hours: int = 24

    
    model_path: str = "ml/skin_model.keras"
    se_reduction_ratio: int = 16
    gradcam_target_layer: str = "multiply"
    low_confidence_threshold: float = 0.50
    very_low_confidence_threshold: float = 0.30

   
    max_upload_mb: int = 10
    min_image_dimension: int = 64
    max_image_dimension: int = 4096

   
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    login_rate_limit_max: int = 5
    login_rate_limit_window_seconds: int = 900

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    
    return Settings()
