"""Detection Protocol

교체 가능한 텍스트 탐지 구현을 위한 인터페이스 정의.
모든 좌표는 원본 이미지 기준 절대 좌표(px).
"""

from typing import Protocol

from src.services.detection.schemas import DetectionResult


class Detector(Protocol):
    """텍스트/말풍선 탐지 인터페이스

    구현체:
    - HFSpaceDetection: HuggingFace Space API
    """

    def detect(self, image_path: str) -> DetectionResult:
        """이미지에서 텍스트/말풍선 영역 탐지

        Args:
            image_path: 이미지 파일 경로

        Returns:
            DetectionResult: 탐지 결과 (bubbles, texts bbox - 원본 이미지 기준 px)

        Raises:
            RuntimeError: API 호출 실패 시
        """
        ...
