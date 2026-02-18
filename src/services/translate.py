"""Translate 서비스: 번역 작업 메타데이터 관리

비즈니스 로직만 담당. Task 호출은 route에서 처리.
"""

import json
import uuid
from datetime import UTC, datetime
from typing import Literal, cast

from pydantic import BaseModel

from src.constants import TTL, RedisPrefix, TranslateId
from src.infra.redis import get_redis
from src.schemas.base import BaseSchema
from src.services.upload import get_upload

TranslateStatus = Literal["pending", "processing", "completed", "failed"]


class TranslateMetadata(BaseModel):
    """Redis에 저장되는 번역 작업 메타데이터"""

    translate_id: str
    status: TranslateStatus
    upload_id: str
    source_language: str
    target_language: str
    created_at: str
    completed_at: str | None = None
    original_url: str | None = None
    result_url: str | None = None
    error_message: str | None = None


class TranslateRequest(BaseSchema):
    """번역 요청"""

    upload_id: str
    source_language: str = "ko"
    target_language: str = "en"


class TranslateResponse(BaseSchema):
    """번역 응답"""

    translate_id: str
    status: TranslateStatus
    upload_id: str
    source_language: str
    target_language: str
    original_url: str | None = None
    result_url: str | None = None
    created_at: str
    completed_at: str | None = None
    error_message: str | None = None


# --- 수정 기능 스키마 (TODO: 구현 예정) ---


class EraseRequest(BaseSchema):
    """영역 지우기 요청

    브러시로 마킹한 영역을 LaMa inpainting으로 제거.
    """

    mask_image: str  # base64 PNG


class FixRequest(BaseSchema):
    """영역 재번역 요청

    번호가 표시된 마킹 이미지 + 텍스트 배열.
    Gemini가 마킹된 영역에 텍스트를 렌더링.
    """

    mask_image: str  # base64 PNG (번호 마킹)
    texts: list[str]  # 번호 순서대로 텍스트


class InvalidUploadError(Exception):
    def __init__(self, upload_id: str):
        self.upload_id = upload_id
        super().__init__(f"존재하지 않는 업로드 ID: {upload_id}")


class TranslateNotFoundError(Exception):
    def __init__(self, translate_id: str):
        self.translate_id = translate_id
        super().__init__(f"존재하지 않는 번역 ID: {translate_id}")


def _generate_translate_id() -> str:
    return f"{TranslateId.PREFIX}{uuid.uuid4().hex[:8]}"


async def validate_upload_id(upload_id: str) -> str:
    """업로드 ID 검증 후 원본 이미지 URL 반환"""
    upload = await get_upload(upload_id)
    if upload is None:
        raise InvalidUploadError(upload_id)
    return upload.image_url


async def update_translate_status(
    translate_id: str,
    status: TranslateStatus,
    error_message: str | None = None,
) -> None:
    """번역 작업 상태 업데이트

    Raises:
        TranslateNotFoundError: 존재하지 않는 번역 ID
    """
    redis = get_redis()
    key = f"{RedisPrefix.TRANSLATE}:{translate_id}"

    data = redis.get(key)
    if data is None:
        raise TranslateNotFoundError(translate_id)

    metadata = TranslateMetadata.model_validate(json.loads(cast(str, data)))
    metadata.status = status

    if error_message is not None:
        metadata.error_message = error_message

    if status == "completed":
        metadata.completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    redis.set(key, metadata.model_dump_json(), keepttl=True)


async def create_translate(request: TranslateRequest, original_url: str) -> TranslateResponse:
    """번역 작업 생성 (메타데이터만, 검증/쿼터/task 호출은 route에서)"""
    redis = get_redis()

    translate_id = _generate_translate_id()
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    metadata = TranslateMetadata(
        translate_id=translate_id,
        status="pending",
        upload_id=request.upload_id,
        source_language=request.source_language,
        target_language=request.target_language,
        original_url=original_url,
        created_at=created_at,
    )

    redis.set(
        f"{RedisPrefix.TRANSLATE}:{translate_id}",
        metadata.model_dump_json(),
        ex=TTL.DATA,
    )

    return TranslateResponse(
        translate_id=metadata.translate_id,
        status=metadata.status,
        upload_id=metadata.upload_id,
        source_language=metadata.source_language,
        target_language=metadata.target_language,
        original_url=metadata.original_url,
        created_at=metadata.created_at,
    )


async def get_translate(translate_id: str) -> TranslateResponse | None:
    """번역 작업 조회"""
    redis = get_redis()

    data = redis.get(f"{RedisPrefix.TRANSLATE}:{translate_id}")
    if data is None:
        return None

    metadata = TranslateMetadata.model_validate(json.loads(cast(str, data)))

    return TranslateResponse(
        translate_id=metadata.translate_id,
        status=metadata.status,
        upload_id=metadata.upload_id,
        source_language=metadata.source_language,
        target_language=metadata.target_language,
        original_url=metadata.original_url,
        result_url=metadata.result_url,
        created_at=metadata.created_at,
        completed_at=metadata.completed_at,
        error_message=metadata.error_message,
    )
