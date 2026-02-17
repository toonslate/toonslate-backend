"""Translation 팩토리 테스트"""

from unittest.mock import patch

import pytest

from src.schemas.pipeline import BBox, TranslationResult
from src.services.translation import get_translation, set_translation
from src.services.translation.gemini import GeminiTranslation


class TestGetTranslation:
    def setup_method(self) -> None:
        set_translation(None)

    def test_default_returns_gemini_translation(self) -> None:
        translator = get_translation()
        assert isinstance(translator, GeminiTranslation)

    def test_get_translation_returns_cached_instance(self) -> None:
        first = get_translation()
        second = get_translation()
        assert first is second

    def test_set_translation_overrides_factory(self) -> None:
        mock = MockTranslator()
        set_translation(mock)
        assert get_translation() is mock

    def test_set_translation_none_resets(self) -> None:
        mock = MockTranslator()
        set_translation(mock)
        set_translation(None)

        translator = get_translation()
        assert isinstance(translator, GeminiTranslation)

    def test_unknown_provider_raises(self) -> None:
        with patch("src.services.translation.get_settings") as mock_settings:
            mock_settings.return_value.translation_provider = "unknown"
            with pytest.raises(ValueError, match="Unknown translation provider"):
                get_translation()


class MockTranslator:
    def translate(self, image_path: str, bboxes: list[BBox]) -> list[TranslationResult]:
        return []
