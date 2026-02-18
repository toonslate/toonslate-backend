"""배치 서비스: 다중 이미지 번역 배치 메타데이터 관리

개별 translate는 translate 서비스를 재사용.
배치 상태는 개별 translate 상태로부터 동적 계산.

TODO: 스키마(Request/Response/Metadata)를 src/schemas/로 분리 검토.
      translate.py도 동일 구조이므로 함께 정리.
"""

import json
import uuid
from datetime import UTC, datetime
from typing import Literal, cast

from pydantic import BaseModel, field_validator

from src.constants import TTL, BatchId, Limits, RedisPrefix
from src.infra.redis import get_redis
from src.schemas.base import BaseSchema
from src.services.translate import (
    TranslateRequest,
    TranslateStatus,
    create_translate,
    get_translate,
)

BatchStatus = Literal["processing", "completed", "partial_failure", "failed"]


class BatchImageEntry(BaseSchema):
    order_index: int
    upload_id: str
    translate_id: str
    status: TranslateStatus
    original_url: str | None = None
    result_url: str | None = None
    error_message: str | None = None


class BatchMetadata(BaseModel):
    batch_id: str
    source_language: str
    target_language: str
    images: list[BatchImageEntry]
    created_at: str


class BatchRequest(BaseSchema):
    upload_ids: list[str]
    source_language: str = "ko"
    target_language: str = "en"

    @field_validator("upload_ids")
    @classmethod
    def validate_upload_ids_length(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("최소 1개의 upload_id가 필요합니다")
        if len(v) > Limits.MAX_BATCH_SIZE:
            raise ValueError(f"최대 {Limits.MAX_BATCH_SIZE}개까지 가능합니다")
        return v


class BatchResponse(BaseSchema):
    batch_id: str
    status: BatchStatus
    images: list[BatchImageEntry]
    source_language: str
    target_language: str
    created_at: str


def _generate_batch_id() -> str:
    return f"{BatchId.PREFIX}{uuid.uuid4().hex[:8]}"


def _compute_batch_status(images: list[BatchImageEntry]) -> BatchStatus:
    statuses = {img.status for img in images}

    if statuses & {"pending", "processing"}:
        return "processing"
    if statuses == {"completed"}:
        return "completed"
    if statuses == {"failed"}:
        return "failed"
    return "partial_failure"


async def create_batch(request: BatchRequest, original_urls: list[str]) -> BatchResponse:
    """배치 생성: 개별 translate 메타데이터 + 배치 래퍼 저장"""
    if len(request.upload_ids) != len(original_urls):
        raise ValueError(
            f"upload_ids와 original_urls의 길이가 다릅니다: "
            f"{len(request.upload_ids)} != {len(original_urls)}"
        )

    redis = get_redis()
    batch_id = _generate_batch_id()
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    images: list[BatchImageEntry] = []

    pairs = zip(request.upload_ids, original_urls, strict=True)
    for i, (upload_id, original_url) in enumerate(pairs):
        translate_request = TranslateRequest(
            upload_id=upload_id,
            source_language=request.source_language,
            target_language=request.target_language,
        )
        translate_response = await create_translate(translate_request, original_url)

        images.append(
            BatchImageEntry(
                order_index=i,
                upload_id=upload_id,
                translate_id=translate_response.translate_id,
                status="pending",
                original_url=original_url,
            )
        )

    metadata = BatchMetadata(
        batch_id=batch_id,
        source_language=request.source_language,
        target_language=request.target_language,
        images=images,
        created_at=created_at,
    )

    redis.set(
        f"{RedisPrefix.BATCH}:{batch_id}",
        metadata.model_dump_json(),
        ex=TTL.DATA,
    )

    return BatchResponse(
        batch_id=batch_id,
        status="processing",
        images=images,
        source_language=request.source_language,
        target_language=request.target_language,
        created_at=created_at,
    )


async def get_batch(batch_id: str) -> BatchResponse | None:
    """배치 조회: 개별 translate 최신 상태 반영 + 배치 상태 동적 계산"""
    redis = get_redis()

    data = redis.get(f"{RedisPrefix.BATCH}:{batch_id}")
    if data is None:
        return None

    metadata = BatchMetadata.model_validate(json.loads(cast(str, data)))

    updated_images: list[BatchImageEntry] = []
    for image in metadata.images:
        translate = await get_translate(image.translate_id)
        if translate is None:
            updated_images.append(
                image.model_copy(
                    update={
                        "status": "failed",
                        "error_message": "번역 메타데이터를 찾을 수 없습니다",
                    }
                )
            )
            continue

        updated_images.append(
            BatchImageEntry(
                order_index=image.order_index,
                upload_id=image.upload_id,
                translate_id=image.translate_id,
                status=translate.status,
                original_url=translate.original_url,
                result_url=translate.result_url,
                error_message=translate.error_message,
            )
        )

    return BatchResponse(
        batch_id=metadata.batch_id,
        status=_compute_batch_status(updated_images),
        images=updated_images,
        source_language=metadata.source_language,
        target_language=metadata.target_language,
        created_at=metadata.created_at,
    )
