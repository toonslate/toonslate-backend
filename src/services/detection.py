# pyright: reportMissingTypeStubs=false
# TODO: gradio_client 타입 스텁 개선 검토 (현재 파일 단위 비활성화)

import time
from typing import Any

from gradio_client import Client, handle_file
from pydantic import BaseModel

from src.config import get_settings


class ImageSize(BaseModel):
    width: int
    height: int


class DetectionResult(BaseModel):
    """HuggingFace Space 탐지 결과"""

    image_size: ImageSize
    bubbles: list[list[float]]  # [[x1, y1, x2, y2], ...]
    bubble_confs: list[float]
    texts: list[list[float]]
    text_confs: list[float]


def detect_regions(image_path: str, max_retries: int = 3) -> DetectionResult:
    """HuggingFace Space API 호출 (재시도 포함)

    Note: HF Space는 슬립 상태일 수 있음. 첫 호출 시 웜업 필요.

    Args:
        image_path: 이미지 파일 경로
        max_retries: 최대 재시도 횟수

    Returns:
        DetectionResult: 탐지 결과 (bubbles, texts bbox)

    Raises:
        RuntimeError: API 호출 실패 시
    """
    settings = get_settings()
    client = Client(settings.hf_space_url, httpx_kwargs={"timeout": settings.hf_api_timeout})

    for attempt in range(max_retries):
        try:
            result: Any = client.predict(  # pyright: ignore[reportUnknownMemberType]
                handle_file(image_path), api_name="/detect"
            )
            return DetectionResult.model_validate(result)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Detection API 호출 실패: {e}") from e

    raise RuntimeError("Detection API 호출 실패: max_retries 초과")
