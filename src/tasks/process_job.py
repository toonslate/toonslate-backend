import logging

from celery import Task

from src.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def process_job(self: Task, job_id: str) -> dict[str, str]:  # type: ignore[type-arg]
    """작업 처리 태스크 (Day 5-7에서 AI 파이프라인 추가 예정)"""
    logger.info(f"Processing job: {job_id}")

    # TODO: JobService.update_job_status(job_id, "processing")
    # TODO: AI 파이프라인 실행 (OCR → 번역 → 이미지 합성)
    # TODO: JobService.update_job_status(job_id, "completed", result_url=...)

    return {"job_id": job_id, "status": "completed"}
