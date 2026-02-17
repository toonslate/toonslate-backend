"""GeminiTranslation 구현체 테스트"""

from unittest.mock import MagicMock, patch

import pytest

from src.schemas.pipeline import BBox, TranslationResult
from src.services.translation.base import TranslationError
from src.services.translation.gemini import GeminiTranslation

GEMINI_MODULE = "src.services.translation.gemini"

VALID_BBOXES = [
    BBox(x1=10, y1=20, x2=100, y2=80),
    BBox(x1=200, y1=300, x2=400, y2=500),
]

MIXED_BBOXES = [
    BBox(x1=10, y1=20, x2=100, y2=80),
    BBox(x1=50, y1=50, x2=50, y2=100),  # width=0, invalid
    BBox(x1=200, y1=300, x2=400, y2=500),
]

MOCK_RESPONSE = '[{"index": 0, "translated": "Hello"}, {"index": 1, "translated": "BOOM"}]'
MOCK_MIXED_RESPONSE = '[{"index": 0, "translated": "First"}, {"index": 1, "translated": "Third"}]'


@patch(f"{GEMINI_MODULE}.types")
@patch(f"{GEMINI_MODULE}.Image")
class TestGeminiTranslation:
    def setup_method(self) -> None:
        self.translator = GeminiTranslation(api_key="test-key", model="test-model")

    def test_translate_returns_results(
        self, _mock_image: MagicMock, _mock_types: MagicMock
    ) -> None:
        with patch(f"{GEMINI_MODULE}.genai") as mock_genai:
            mock_response = MagicMock()
            mock_response.text = MOCK_RESPONSE
            mock_genai.Client.return_value.models.generate_content.return_value = mock_response

            results = self.translator.translate("test.png", VALID_BBOXES)

        assert len(results) == 2
        assert results[0] == TranslationResult(index=0, translated="Hello")
        assert results[1] == TranslationResult(index=1, translated="BOOM")

    def test_translate_empty_bboxes(self, _mock_image: MagicMock, _mock_types: MagicMock) -> None:
        results = self.translator.translate("test.png", [])
        assert results == []

    def test_translate_skips_invalid_bbox(
        self, _mock_image: MagicMock, _mock_types: MagicMock
    ) -> None:
        with patch(f"{GEMINI_MODULE}.genai") as mock_genai:
            mock_response = MagicMock()
            mock_response.text = MOCK_MIXED_RESPONSE
            mock_genai.Client.return_value.models.generate_content.return_value = mock_response

            results = self.translator.translate("test.png", MIXED_BBOXES)

        assert len(results) == 2
        assert results[0] == TranslationResult(index=0, translated="First")
        assert results[1] == TranslationResult(index=2, translated="Third")

    def test_translate_no_api_key_raises(
        self, _mock_image: MagicMock, _mock_types: MagicMock
    ) -> None:
        translator = GeminiTranslation(api_key="", model="test-model")
        with pytest.raises(TranslationError):
            translator.translate("test.png", VALID_BBOXES)

    def test_translate_empty_response_raises(
        self, _mock_image: MagicMock, _mock_types: MagicMock
    ) -> None:
        with patch(f"{GEMINI_MODULE}.genai") as mock_genai:
            mock_response = MagicMock()
            mock_response.text = None
            mock_genai.Client.return_value.models.generate_content.return_value = mock_response

            with pytest.raises(TranslationError):
                self.translator.translate("test.png", VALID_BBOXES)

    def test_translate_json_parse_failure_raises(
        self, _mock_image: MagicMock, _mock_types: MagicMock
    ) -> None:
        with patch(f"{GEMINI_MODULE}.genai") as mock_genai:
            mock_response = MagicMock()
            mock_response.text = "not valid json {"
            mock_genai.Client.return_value.models.generate_content.return_value = mock_response

            with pytest.raises(TranslationError):
                self.translator.translate("test.png", VALID_BBOXES)

    def test_translate_non_list_response_raises(
        self, _mock_image: MagicMock, _mock_types: MagicMock
    ) -> None:
        with patch(f"{GEMINI_MODULE}.genai") as mock_genai:
            mock_response = MagicMock()
            mock_response.text = '{"not": "a list"}'
            mock_genai.Client.return_value.models.generate_content.return_value = mock_response

            with pytest.raises(TranslationError):
                self.translator.translate("test.png", VALID_BBOXES)
