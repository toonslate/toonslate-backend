from celery import Celery

from src.config import get_settings
from src.constants import TTL

settings = get_settings()

celery_app = Celery(
    "toonslate",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    result_expires=TTL.CELERY_RESULT,
    imports=["src.infra.workers.translate_job"],
    worker_prefetch_multiplier=1,
)
