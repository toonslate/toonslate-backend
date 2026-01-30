import json
import logging
from pathlib import Path
from typing import Any, cast

import cv2
from celery.exceptions import SoftTimeLimitExceeded

from src.celery_app import celery_app
from src.config import get_settings
from src.constants import RedisPrefix
from src.infra.redis import get_redis
from src.infra.storage import get_storage
from src.schemas.pipeline import BBox, TextRegion
from src.services.detection import detect_regions
from src.services.inpainting import get_inpainting
from src.services.job_status import update_job_status
from src.services.rendering import render_translations
from src.services.translation import translate_regions

logger = logging.getLogger(__name__)


def _get_image_path(job_id: str) -> str | None:
    """job_id로부터 첫 번째 이미지의 절대 경로 반환"""
    redis = get_redis()

    job_data = redis.get(f"{RedisPrefix.JOB}:{job_id}")
    if not job_data:
        return None

    job = json.loads(cast(str, job_data))
    upload_ids: list[str] = job.get("upload_ids", [])
    if not upload_ids:
        return None

    upload_data = redis.get(f"{RedisPrefix.UPLOAD}:{upload_ids[0]}")
    if not upload_data:
        return None

    upload = json.loads(cast(str, upload_data))
    relative_path: str = upload.get("path", "")
    if not relative_path:
        return None

    storage = get_storage()
    if not storage.exists(relative_path):
        return None

    return storage.get_absolute_path(relative_path)


def _process_job_sync(job_id: str) -> dict[str, Any]:
    """동기 작업 처리

    Pipeline: Detection → Translation → Inpainting → Rendering
    """
    try:
        update_job_status(job_id, "processing")

        image_path = _get_image_path(job_id)
        if not image_path:
            update_job_status(job_id, "failed", error_message="이미지를 찾을 수 없음")
            return {"job_id": job_id, "status": "failed", "error": "이미지를 찾을 수 없음"}

        # 1. Detection
        logger.info(f"[{job_id}] Detection 시작: {image_path}")
        detection_result = detect_regions(image_path)
        logger.info(
            f"[{job_id}] Detection 완료: bubbles={len(detection_result.bubbles)}, "
            f"texts={len(detection_result.texts)}"
        )

        if not detection_result.texts:
            update_job_status(job_id, "completed")
            return {
                "job_id": job_id,
                "status": "completed",
                "message": "텍스트 영역이 감지되지 않음",
            }

        # Detection 결과를 스키마 형태로 변환
        text_bboxes = [BBox.from_list(coords) for coords in detection_result.texts]
        bubble_bboxes = [BBox.from_list(coords) for coords in detection_result.bubbles]

        # 2. Translation (Gemini OCR + 번역)
        logger.info(f"[{job_id}] Translation 시작: {len(text_bboxes)}개 영역")
        translations = translate_regions(image_path, text_bboxes)
        logger.info(f"[{job_id}] Translation 완료: {len(translations)}개 번역")

        if not translations:
            update_job_status(job_id, "completed")
            return {
                "job_id": job_id,
                "status": "completed",
                "message": "번역 가능한 텍스트 없음",
            }

        # TextRegion 생성 (index 포함)
        text_regions = [TextRegion(index=i, text_bbox=bbox) for i, bbox in enumerate(text_bboxes)]

        # 3. Inpainting (텍스트 영역 지우기)
        logger.info(f"[{job_id}] Inpainting 시작")
        image = cv2.imread(image_path)
        if image is None:
            update_job_status(job_id, "failed", error_message="이미지 로드 실패")
            return {"job_id": job_id, "status": "failed", "error": "이미지 로드 실패"}

        # BGR 3채널 검증
        if len(image.shape) != 3 or image.shape[2] != 3:
            update_job_status(job_id, "failed", error_message="지원하지 않는 이미지 형식")
            return {"job_id": job_id, "status": "failed", "error": "BGR 3채널 이미지만 지원"}

        inpainting = get_inpainting()
        clean_image, updated_regions = inpainting.inpaint(image, text_regions, bubble_bboxes)
        logger.info(f"[{job_id}] Inpainting 완료")

        # 4. Rendering (번역 텍스트 삽입)
        logger.info(f"[{job_id}] Rendering 시작")
        result_image = render_translations(clean_image, updated_regions, translations)
        logger.info(f"[{job_id}] Rendering 완료")

        # 5. 결과 저장 (job_id 포함하여 경로 고유화)
        storage = get_storage()

        # clean 이미지 저장 (cv2 BGR)
        clean_relative = f"clean/{job_id}_clean.png"
        clean_abs_path = Path(storage.get_absolute_path(clean_relative))
        clean_abs_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(clean_abs_path), clean_image)

        # result 이미지 저장 (PIL RGB)
        result_relative = f"result/{job_id}_result.png"
        result_abs_path = Path(storage.get_absolute_path(result_relative))
        result_abs_path.parent.mkdir(parents=True, exist_ok=True)
        result_image.save(str(result_abs_path))

        logger.info(f"[{job_id}] 결과 저장 완료: {result_relative}")

        settings = get_settings()
        result_url = f"{settings.base_url}/static/{result_relative}"
        update_job_status(job_id, "completed", result_url=result_url)

        return {
            "job_id": job_id,
            "status": "completed",
            "detection": {
                "bubbles": len(detection_result.bubbles),
                "texts": len(detection_result.texts),
            },
            "translations": len(translations),
            "result_path": result_relative,
            "clean_path": clean_relative,
        }

    except SoftTimeLimitExceeded:
        logger.error(f"[{job_id}] 처리 시간 초과")
        update_job_status(job_id, "failed", error_message="처리 시간 초과")
        return {"job_id": job_id, "status": "failed", "error": "timeout"}

    except Exception as e:
        logger.exception(f"[{job_id}] 파이프라인 실패: {e}")
        update_job_status(job_id, "failed", error_message=str(e))
        return {"job_id": job_id, "status": "failed", "error": str(e)}


@celery_app.task(soft_time_limit=300, time_limit=360)
def process_job(job_id: str) -> dict[str, Any]:
    """작업 처리 태스크

    Timeout:
        - soft_time_limit=300: 5분 (SoftTimeLimitExceeded 발생)
        - time_limit=360: 6분 (강제 종료)
        - 개별 API 호출은 자체 타임아웃 설정 필요 (soft_time_limit보다 짧게)
    """
    logger.info(f"Processing job: {job_id}")
    return _process_job_sync(job_id)
