"""Batch API 라우트

다중 이미지 배치 번역 엔드포인트.

NOTE: 오케스트레이션(service 호출 + task 트리거)을 Route에서 처리.
TODO: 규모 확장 시 Use Case 레이어로 분리 검토 (#25).
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request, status

from src.infra.workers.translate_job import translate_job
from src.services import batch as batch_service
from src.services import translate as translate_service
from src.services.quota import (
    QuotaExceededError,
    check_and_consume_quota,
    hash_ip,
    refund_quota,
)

router = APIRouter(prefix="/batch", tags=["batch"])
logger = logging.getLogger(__name__)


def _get_client_ip(req: Request) -> str:
    if req.client:
        return req.client.host

    logger.warning("클라이언트 IP를 확인할 수 없음")
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"code": "UNKNOWN_CLIENT", "message": "클라이언트 IP를 확인할 수 없습니다"},
    )


@router.post(
    "",
    response_model=batch_service.BatchResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_batch(
    request: batch_service.BatchRequest,
    req: Request,
) -> batch_service.BatchResponse:
    """배치 번역 작업 생성"""
    hashed_ip = hash_ip(_get_client_ip(req))
    image_count = len(request.upload_ids)

    original_urls: list[str] = []
    for upload_id in request.upload_ids:
        try:
            url = await translate_service.validate_upload_id(upload_id)
            original_urls.append(url)
        except translate_service.InvalidUploadError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "INVALID_UPLOAD_ID",
                    "message": f"유효하지 않은 업로드 ID: {e.upload_id}",
                },
            ) from None

    try:
        await check_and_consume_quota(hashed_ip, image_count)
    except QuotaExceededError:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": "RATE_LIMIT_EXCEEDED", "message": "주간 사용량 한도를 초과했습니다"},
        ) from None

    try:
        response = await batch_service.create_batch(request, original_urls)
    except Exception:
        try:
            await refund_quota(hashed_ip, image_count)
        except Exception:
            logger.error("쿼터 환급 실패")
        raise

    failed_count = 0
    for image in response.images:
        try:
            await asyncio.to_thread(translate_job.delay, image.translate_id)
        except Exception:
            logger.error(f"Celery 큐잉 실패: {image.translate_id}")
            failed_count += 1
            try:
                await translate_service.update_translate_status(
                    image.translate_id,
                    "failed",
                    "작업 큐잉에 실패했습니다.",
                )
            except Exception:
                logger.error(f"상태 업데이트 실패: {image.translate_id}")

    if failed_count == image_count:
        try:
            await refund_quota(hashed_ip, image_count)
        except Exception:
            logger.error("쿼터 전체 환급 실패")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "QUEUE_UNAVAILABLE",
                "message": "작업 큐가 일시적으로 사용할 수 없습니다",
            },
        )

    if failed_count > 0:
        try:
            await refund_quota(hashed_ip, failed_count)
        except Exception:
            logger.error("쿼터 부분 환급 실패")

    return response


@router.get("/{batch_id}", response_model=batch_service.BatchResponse)
async def get_batch(batch_id: str) -> batch_service.BatchResponse:
    """배치 번역 작업 상태/결과 조회"""
    result = await batch_service.get_batch(batch_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "BATCH_NOT_FOUND",
                "message": f"배치 작업을 찾을 수 없습니다: {batch_id}",
            },
        )

    return result
