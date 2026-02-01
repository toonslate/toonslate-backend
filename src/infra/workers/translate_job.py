"""Nano Banana 번역 작업 처리

Celery 워커에서 실행되는 백그라운드 태스크.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from celery.exceptions import SoftTimeLimitExceeded

from src.config import get_settings
from src.constants import RedisPrefix
from src.infra.celery_app import celery_app
from src.infra.redis import get_redis
from src.infra.storage import get_storage
from src.services.nano_banana import NanoBananaError, translate_image

logger = logging.getLogger(__name__)


def _get_image_path(translate_id: str) -> str | None:
    """translate_id → upload_id → 파일 경로 체이닝 조회"""
    redis = get_redis()

    translate_data = redis.get(f"{RedisPrefix.TRANSLATE}:{translate_id}")
    if not translate_data:
        return None

    metadata = json.loads(cast(str, translate_data))
    upload_id = metadata.get("upload_id")
    if not upload_id:
        return None

    upload_data = redis.get(f"{RedisPrefix.UPLOAD}:{upload_id}")
    if not upload_data:
        return None

    upload = json.loads(cast(str, upload_data))
    relative_path = upload.get("path")
    if not relative_path:
        return None

    storage = get_storage()
    if not storage.exists(relative_path):
        return None

    return storage.get_absolute_path(relative_path)


def _update_status(
    translate_id: str,
    status: str,
    result_url: str | None = None,
    error_message: str | None = None,
) -> None:
    """Redis에 번역 작업 상태 업데이트 (FE 폴링용)"""
    redis = get_redis()
    key = f"{RedisPrefix.TRANSLATE}:{translate_id}"

    data = redis.get(key)
    if not data:
        return

    metadata = json.loads(cast(str, data))
    metadata["status"] = status

    if result_url:
        metadata["result_url"] = result_url

    if error_message:
        metadata["error_message"] = error_message

    if status == "completed":
        metadata["completed_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    redis.set(key, json.dumps(metadata), keepttl=True)


@celery_app.task(soft_time_limit=300, time_limit=360)
def translate_job(translate_id: str) -> dict[str, Any]:
    """번역 작업 태스크

    Celery 워커에서 동기적으로 실행됨.

    Timeout:
        - soft_time_limit=300: 5분 (SoftTimeLimitExceeded 발생)
        - time_limit=360: 6분 (강제 종료)
    """
    logger.info(f"[{translate_id}] 번역 시작")

    try:
        _update_status(translate_id, "processing")

        image_path = _get_image_path(translate_id)
        if not image_path:
            _update_status(translate_id, "failed", error_message="이미지를 찾을 수 없음")
            return {"status": "failed", "error": "이미지를 찾을 수 없음"}

        result_image = translate_image(image_path)

        storage = get_storage()
        result_relative = f"result/{translate_id}_result.png"
        result_abs_path = Path(storage.get_absolute_path(result_relative))
        result_abs_path.parent.mkdir(parents=True, exist_ok=True)
        result_image.save(str(result_abs_path))

        settings = get_settings()
        result_url = f"{settings.base_url}/static/{result_relative}"
        _update_status(translate_id, "completed", result_url=result_url)

        logger.info(f"[{translate_id}] 번역 완료: {result_relative}")
        return {"status": "completed", "result_url": result_url}

    except NanoBananaError as e:
        logger.error(f"[{translate_id}] Nano Banana 오류: {e}")
        _update_status(translate_id, "failed", error_message=str(e))
        return {"status": "failed", "error": str(e)}

    except SoftTimeLimitExceeded:
        logger.error(f"[{translate_id}] 시간 초과")
        _update_status(translate_id, "failed", error_message="처리 시간 초과")
        return {"status": "failed", "error": "timeout"}

    except Exception as e:
        logger.exception(f"[{translate_id}] 예외 발생: {e}")
        _update_status(translate_id, "failed", error_message=str(e))
        return {"status": "failed", "error": str(e)}
