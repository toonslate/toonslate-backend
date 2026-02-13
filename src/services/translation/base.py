"""Translation Protocol

교체 가능한 번역 구현을 위한 인터페이스 정의.
"""

from typing import Protocol

from src.schemas.pipeline import BBox, TranslationResult


class TranslationError(Exception):
    pass


class Translator(Protocol):
    """텍스트 번역 인터페이스

    구현체:
    - GeminiTranslation: Google Gemini API
    """

    def translate(self, image_path: str, bboxes: list[BBox]) -> list[TranslationResult]:
        """텍스트 영역들을 번역

        Args:
            image_path: 이미지 파일 경로
            bboxes: 텍스트 영역 바운딩 박스 리스트

        Returns:
            list[TranslationResult]: 번역 결과 (원본 bbox 인덱스 기준)

        Raises:
            TranslationError: 번역 실패 시
        """
        ...
