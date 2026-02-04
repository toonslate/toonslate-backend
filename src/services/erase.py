"""Erase 서비스: 브러시 마킹 영역 inpainting 제거

FE에서 캔버스로 마킹한 영역을 마스크로 받아서 제거합니다.
Inpainting 백엔드는 환경변수 INPAINTING_PROVIDER로 선택 (iopaint_lama, replicate_lama, solid_fill)

TODO: 프로젝트 전체 async/sync 일관성 검토 (storage/local.py 등 동기 I/O 사용 중)
TODO: 에러 메시지 중앙화 - ErrorCode enum 도입 검토 (현재 EraseError.code로 문자열 관리)
TODO: 동시 요청 증가 시 threadpool 고갈 위험 - 작업 큐(Celery) 검토
TODO: 현재 LocalStorage 전용 - S3 등 원격 스토리지 사용 시 get_absolute_path 대신 임시 다운로드 필요
TODO: 마스크 입력 오류(invalid base64)는 현재 500/INPAINTING_FAILED
      이후 400/INVALID_MASK_IMAGE로 분리 예정
TODO: 말풍선(흰색 배경) 텍스트는 LaMa보다 단순 흰색 채우기가 더 적합
      향후 말풍선 감지 → 흰색 채우기 / 그 외 → LaMa 분기 처리 검토
"""

import base64
import io
import json
import logging
from typing import cast

import cv2
import numpy as np
from PIL import Image

from src.constants import RedisPrefix, TranslateId
from src.infra.redis import get_redis
from src.infra.storage import get_storage
from src.schemas.base import BaseSchema
from src.services.inpainting import get_inpainting

logger = logging.getLogger(__name__)


class EraseError(Exception):
    """Erase 작업 관련 에러

    code로 구체적인 원인 구분:
    - TRANSLATE_NOT_FOUND: 번역 메타데이터 없음 (404)
    - TRANSLATE_NOT_COMPLETED: 번역 미완료 (400)
    - RESULT_IMAGE_NOT_FOUND: 결과 이미지 파일 없음 (404)
    - INPAINTING_FAILED: inpainting 실패 (500)
    """

    STATUS_MAP: dict[str, int] = {
        "INVALID_TRANSLATE_ID": 400,
        "TRANSLATE_NOT_FOUND": 404,
        "TRANSLATE_NOT_COMPLETED": 400,
        "RESULT_IMAGE_NOT_FOUND": 404,
        "INPAINTING_FAILED": 500,
    }

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")

    @property
    def status_code(self) -> int:
        return self.STATUS_MAP.get(self.code, 500)


class EraseRequest(BaseSchema):
    """영역 지우기 요청"""

    translate_id: str
    mask_image: str  # base64 PNG


class EraseResponse(BaseSchema):
    """영역 지우기 응답"""

    result_image: str  # base64 PNG


def _b64_to_numpy(b64_str: str) -> np.ndarray:
    """base64 PNG → numpy 배열

    Raises:
        EraseError: 디코딩 또는 이미지 파싱 실패
    """
    try:
        img_data = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_data))
        return np.array(img)
    except Exception as e:
        raise EraseError("INPAINTING_FAILED", "마스크 이미지 디코딩 실패") from e


def _numpy_to_b64(arr: np.ndarray) -> str:
    """numpy 배열 → base64 PNG"""
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _validate_translate_id(translate_id: str) -> None:
    """translate_id 형식 검증 (Path Traversal 방지)

    Raises:
        EraseError: 형식이 올바르지 않음
    """
    if not TranslateId.PATTERN.match(translate_id):
        raise EraseError("INVALID_TRANSLATE_ID", f"올바르지 않은 번역 ID 형식: {translate_id}")


def _get_result_image_path(translate_id: str) -> str:
    """translate_id → 번역 결과 이미지 절대 경로

    Raises:
        EraseError: 형식 오류 / 번역 없음 / 미완료 / 파일 없음
    """
    _validate_translate_id(translate_id)

    redis = get_redis()
    storage = get_storage()

    translate_data = redis.get(f"{RedisPrefix.TRANSLATE}:{translate_id}")
    if not translate_data:
        raise EraseError("TRANSLATE_NOT_FOUND", f"번역을 찾을 수 없습니다: {translate_id}")

    try:
        metadata = json.loads(cast(str, translate_data))
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as e:
        logger.error(f"Redis 데이터 파싱 실패: {translate_id} - {e}")
        raise EraseError("INPAINTING_FAILED", "번역 메타데이터 파싱 실패") from e

    status = metadata.get("status", "unknown")
    if status != "completed":
        raise EraseError("TRANSLATE_NOT_COMPLETED", f"번역이 완료되지 않았습니다 (현재: {status})")

    result_relative = f"result/{translate_id}_result.png"
    if not storage.exists(result_relative):
        raise EraseError("RESULT_IMAGE_NOT_FOUND", "번역 결과 이미지 파일이 없습니다")

    return storage.get_absolute_path(result_relative)


def ensure_grayscale_mask(mask: np.ndarray) -> np.ndarray:
    """마스크를 그레이스케일로 변환"""
    if len(mask.shape) == 2:
        return mask
    if len(mask.shape) == 3:
        if mask.shape[2] == 1:
            return mask[:, :, 0]
        if mask.shape[2] == 4:
            return cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)
        if mask.shape[2] == 3:
            return cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    raise EraseError("INPAINTING_FAILED", f"지원하지 않는 마스크 형식: {mask.shape}")


def erase_region(request: EraseRequest) -> EraseResponse:
    """브러시 마킹 영역 제거

    동기 함수 - FastAPI가 threadpool에서 실행.

    Raises:
        EraseError: 모든 에러 (code로 구분)
    """
    image_path = _get_result_image_path(request.translate_id)

    img = cv2.imread(image_path)
    if img is None:
        raise EraseError("INPAINTING_FAILED", f"이미지 로드 실패: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = _b64_to_numpy(request.mask_image)
    mask = ensure_grayscale_mask(mask)

    # 0이 아닌 픽셀을 255로 (IOPaint는 흰색이 마스크 영역)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    if img_rgb.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(
            mask,
            (img_rgb.shape[1], img_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    try:
        backend = get_inpainting()
        result_rgb = backend.inpaint_mask(img_rgb, mask)
    except Exception as e:
        logger.error(f"Inpainting 실패: {e}")
        raise EraseError("INPAINTING_FAILED", f"Inpainting 실패: {e}") from e

    result_b64 = _numpy_to_b64(result_rgb)

    logger.info(f"[{request.translate_id}] Erase 완료")

    return EraseResponse(result_image=result_b64)
