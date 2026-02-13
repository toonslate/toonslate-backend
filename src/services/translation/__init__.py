"""Translation 모듈

사용법:
    from src.services.translation import get_translation

    translator = get_translation()
    results = translator.translate(image_path, bboxes)

백엔드 선택 (.env TRANSLATION_PROVIDER):
    - "gemini": Google Gemini API (기본값)
"""

from src.config import get_settings
from src.services.translation.base import TranslationError, Translator
from src.services.translation.gemini import GeminiTranslation

__all__ = ["Translator", "TranslationError", "get_translation", "set_translation"]

_translator: Translator | None = None


def get_translation() -> Translator:
    """설정에 따라 translation 백엔드 반환"""
    global _translator
    if _translator is None:
        settings = get_settings()
        if settings.translation_provider == "gemini":
            _translator = GeminiTranslation(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
            )
        else:
            raise ValueError(f"Unknown translation provider: {settings.translation_provider!r}")
    return _translator


def set_translation(translator: Translator | None) -> None:
    """translation 백엔드 설정 (테스트용)"""
    global _translator
    _translator = translator
