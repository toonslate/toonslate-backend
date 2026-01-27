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

    # Security
    ip_hash_secret: str = "change-me-in-production"

    # HuggingFace Space
    hf_space_url: str = "lazistar/toonslate-detector"
    hf_api_timeout: int = 120


@lru_cache
def get_settings() -> Settings:
    return Settings()
