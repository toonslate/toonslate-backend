"""Erase API 라우트

브러시로 마킹한 영역을 LaMa inpainting으로 제거하는 엔드포인트.
"""

from fastapi import APIRouter, HTTPException, status

from src.services import erase as erase_service

router = APIRouter(prefix="/erase", tags=["erase"])


@router.post(
    "",
    response_model=erase_service.EraseResponse,
    status_code=status.HTTP_200_OK,
)
def erase(request: erase_service.EraseRequest) -> erase_service.EraseResponse:
    """브러시 마킹 영역 제거

    동기 엔드포인트 - FastAPI가 threadpool에서 실행.
    """
    try:
        return erase_service.erase_region(request)
    except erase_service.EraseError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={"code": e.code, "message": e.message},
        ) from None
