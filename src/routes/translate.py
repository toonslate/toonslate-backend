"""Translate API 라우트

Nano Banana 기반 웹툰 번역 엔드포인트.

NOTE: 오케스트레이션(service 호출 + task 트리거)을 Route에서 처리.
      순환 참조 회피를 위한 MVP 단순화 결정.
TODO: 규모 확장 시 Use Case 레이어로 분리 검토.
      참고: docs/learnings/orchestration-architecture.md
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request, status

from src.infra.workers.translate_job import translate_job
from src.services import translate as translate_service
from src.services.quota import QuotaExceededError, check_and_consume_quota, hash_ip, refund_quota

router = APIRouter(prefix="/translate", tags=["translate"])
logger = logging.getLogger(__name__)


def _get_client_ip(req: Request) -> str:
    """클라이언트 IP 추출

    X-Forwarded-For는 신뢰 프록시 설정 없이 사용하면 쿼터 우회가 가능하므로,
    배포 인프라가 결정되기 전까지 req.client.host만 사용한다.
    """
    if req.client:
        return req.client.host

    logger.warning("클라이언트 IP를 확인할 수 없음")
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"code": "UNKNOWN_CLIENT", "message": "클라이언트 IP를 확인할 수 없습니다"},
    )


@router.post(
    "",
    response_model=translate_service.TranslateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_translate(
    request: translate_service.TranslateRequest,
    req: Request,
) -> translate_service.TranslateResponse:
    """번역 작업 생성"""
    hashed_ip = hash_ip(_get_client_ip(req))

    try:
        original_url = await translate_service.validate_upload_id(request.upload_id)
    except translate_service.InvalidUploadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "INVALID_UPLOAD_ID",
                "message": f"유효하지 않은 업로드 ID: {e.upload_id}",
            },
        ) from None

    try:
        await check_and_consume_quota(hashed_ip, 1)
    except QuotaExceededError:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": "RATE_LIMIT_EXCEEDED", "message": "주간 사용량 한도를 초과했습니다"},
        ) from None

    try:
        response = await translate_service.create_translate(request, original_url)
    except Exception:
        try:
            await refund_quota(hashed_ip, 1)
        except Exception:
            logger.error("쿼터 환급 실패")
        raise

    try:
        await asyncio.to_thread(translate_job.delay, response.translate_id)
    except Exception as e:
        logger.error(f"Celery 큐잉 실패: {e}")
        try:
            await refund_quota(hashed_ip, 1)
        except Exception:
            logger.error("쿼터 환급 실패")
        try:
            await translate_service.update_translate_status(
                response.translate_id,
                "failed",
                "작업 큐잉에 실패했습니다. 잠시 후 다시 시도해주세요.",
            )
        except Exception:
            logger.error(f"상태 업데이트 실패: {response.translate_id}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "QUEUE_UNAVAILABLE",
                "message": "작업 큐잉에 실패했습니다. 잠시 후 다시 시도해주세요.",
                "translate_id": response.translate_id,
            },
        ) from None

    return response


@router.get("/{translate_id}", response_model=translate_service.TranslateResponse)
async def get_translate(translate_id: str) -> translate_service.TranslateResponse:
    """번역 작업 상태/결과 조회"""
    result = await translate_service.get_translate(translate_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "TRANSLATE_NOT_FOUND",
                "message": f"번역 작업을 찾을 수 없습니다: {translate_id}",
            },
        )

    return result
