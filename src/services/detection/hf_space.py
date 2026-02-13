"""HuggingFace Space Detection 구현체"""

# pyright: reportMissingTypeStubs=false

import time
from typing import Any

from gradio_client import Client, handle_file

from src.services.detection.schemas import DetectionResult


class HFSpaceDetection:
    """HuggingFace Space API를 사용한 텍스트/말풍선 탐지

    Note: HF Space는 슬립 상태일 수 있음. 첫 호출 시 웜업 필요.
    """

    def __init__(self, space_url: str, api_timeout: int = 120) -> None:
        self._space_url = space_url
        self._api_timeout = api_timeout

    def detect(self, image_path: str, max_retries: int = 3) -> DetectionResult:
        """이미지에서 텍스트/말풍선 영역 탐지 (재시도 포함)

        Args:
            image_path: 이미지 파일 경로
            max_retries: 최대 재시도 횟수

        Returns:
            DetectionResult: 탐지 결과 (원본 이미지 기준 px 좌표)

        Raises:
            RuntimeError: API 호출 반복 실패 시
            ValidationError: API 응답 스키마 불일치 시 (재시도 없이 즉시)
        """
        client = Client(self._space_url, httpx_kwargs={"timeout": self._api_timeout})
        raw = self._call_with_retry(client, image_path, max_retries)
        return DetectionResult.model_validate(raw)

    def _call_with_retry(self, client: Client, image_path: str, max_retries: int) -> Any:
        last_error: Exception | None = None
        for attempt in range(1 + max_retries):
            try:
                return client.predict(handle_file(image_path), api_name="/detect")
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(2**attempt)

        raise RuntimeError(f"Detection API 호출 실패: {last_error}") from last_error
