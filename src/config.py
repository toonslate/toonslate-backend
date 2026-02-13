from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # App
    base_url: str = "http://localhost:8000"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # CORS
    cors_origins: list[str] = ["http://localhost:5173"]

    # Security
    ip_hash_secret: str = "change-me-in-production"

    # Detection
    detection_provider: str = "hf_space"  # "hf_space"
    hf_space_url: str = "lazistar/toonslate-detector"
    hf_api_timeout: int = 120

    # Gemini API
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"

    # Replicate API
    replicate_api_token: str = ""

    # Inpainting
    inpainting_provider: str = "solid_fill"  # "solid_fill" | "replicate_lama" | "iopaint_lama"
    inpainting_debug_dir: str = ""
    iopaint_space_url: str = "https://sanster-iopaint-lama.hf.space"
    iopaint_timeout: int = 120


@lru_cache
def get_settings() -> Settings:
    return Settings()
