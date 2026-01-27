import asyncio
import json
import logging
from typing import Any

from celery.exceptions import SoftTimeLimitExceeded

from src.celery_app import celery_app
from src.constants import RedisPrefix
from src.infra.redis import get_redis
from src.infra.storage import get_storage
from src.services.detection import detect_regions
from src.services.job_status import update_job_status

logger = logging.getLogger(__name__)


async def _get_image_path(job_id: str) -> str | None:
    """job_id로부터 첫 번째 이미지의 절대 경로 반환"""
    redis = get_redis()

    job_data = await redis.get(f"{RedisPrefix.JOB}:{job_id}")
    if not job_data:
        return None

    job = json.loads(job_data)
    upload_ids: list[str] = job.get("upload_ids", [])
    if not upload_ids:
        return None

    upload_data = await redis.get(f"{RedisPrefix.UPLOAD}:{upload_ids[0]}")
    if not upload_data:
        return None

    upload = json.loads(upload_data)
    relative_path: str = upload.get("path", "")
    if not relative_path:
        return None

    storage = get_storage()
    if not storage.exists(relative_path):
        return None

    return storage.get_absolute_path(relative_path)


@celery_app.task(soft_time_limit=300, time_limit=360)
def process_job(job_id: str) -> dict[str, Any]:
    """작업 처리 태스크

    Timeout:
        - soft_time_limit=300: 5분 (SoftTimeLimitExceeded 발생)
        - time_limit=360: 6분 (강제 종료)
        - 개별 API 호출은 자체 타임아웃 설정 필요 (soft_time_limit보다 짧게)
    """
    logger.info(f"Processing job: {job_id}")

    try:
        asyncio.run(update_job_status(job_id, "processing"))

        image_path = asyncio.run(_get_image_path(job_id))
        if not image_path:
            asyncio.run(update_job_status(job_id, "failed", error_message="이미지를 찾을 수 없음"))
            return {"job_id": job_id, "status": "failed"}

        logger.info(f"Running detection for: {image_path}")
        detection_result = detect_regions(image_path)

        logger.info(
            f"Detection complete: {len(detection_result.bubbles)} bubbles, "
            f"{len(detection_result.texts)} texts"
        )

        # TODO: OCR → 번역 → Inpainting → 렌더링 (Day 6 이후)

        asyncio.run(update_job_status(job_id, "completed"))

        return {
            "job_id": job_id,
            "status": "completed",
            "detection": {
                "bubbles": len(detection_result.bubbles),
                "texts": len(detection_result.texts),
            },
        }

    except SoftTimeLimitExceeded:
        logger.error(f"Job {job_id} timed out")
        asyncio.run(update_job_status(job_id, "failed", error_message="처리 시간 초과"))
        return {"job_id": job_id, "status": "failed", "error": "timeout"}

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        asyncio.run(update_job_status(job_id, "failed", error_message=str(e)))
        return {"job_id": job_id, "status": "failed", "error": str(e)}
