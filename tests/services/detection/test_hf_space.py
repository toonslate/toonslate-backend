"""HFSpaceDetection 구현체 테스트"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.services.detection.hf_space import HFSpaceDetection

MOCK_API_RESPONSE = {
    "image_size": {"width": 800, "height": 1200},
    "bubbles": [[10.0, 20.0, 200.0, 100.0], [300.0, 400.0, 500.0, 600.0]],
    "bubble_confs": [0.95, 0.87],
    "texts": [[15.0, 25.0, 190.0, 90.0], [310.0, 410.0, 490.0, 590.0]],
    "text_confs": [0.92, 0.88],
}

MOCK_EMPTY_RESPONSE: dict[str, Any] = {
    "image_size": {"width": 800, "height": 1200},
    "bubbles": [],
    "bubble_confs": [],
    "texts": [],
    "text_confs": [],
}

HF_SPACE_MODULE = "src.services.detection.hf_space"


@patch(f"{HF_SPACE_MODULE}.handle_file", return_value="mock_file_handle")
class TestHFSpaceDetection:
    def setup_method(self) -> None:
        self.detector = HFSpaceDetection(space_url="test/space", api_timeout=10)

    def test_detect_returns_detection_result(self, _mock_handle: MagicMock) -> None:
        with patch(f"{HF_SPACE_MODULE}.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.predict.return_value = MOCK_API_RESPONSE
            mock_client_cls.return_value = mock_client

            result = self.detector.detect("test.png")

        assert result.image_size.width == 800
        assert result.image_size.height == 1200
        assert len(result.bubbles) == 2
        assert len(result.texts) == 2
        assert result.bubbles[0] == [10.0, 20.0, 200.0, 100.0]

    def test_detect_empty_result(self, _mock_handle: MagicMock) -> None:
        with patch(f"{HF_SPACE_MODULE}.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.predict.return_value = MOCK_EMPTY_RESPONSE
            mock_client_cls.return_value = mock_client

            result = self.detector.detect("test.png")

        assert result.bubbles == []
        assert result.texts == []

    @patch(f"{HF_SPACE_MODULE}.time.sleep")
    def test_detect_retries_on_failure(
        self, _mock_sleep: MagicMock, _mock_handle: MagicMock
    ) -> None:
        with patch(f"{HF_SPACE_MODULE}.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.predict.side_effect = [
                Exception("API 일시 장애"),
                MOCK_API_RESPONSE,
            ]
            mock_client_cls.return_value = mock_client

            result = self.detector.detect("test.png")

        assert len(result.bubbles) == 2
        assert mock_client.predict.call_count == 2

    @patch(f"{HF_SPACE_MODULE}.time.sleep")
    def test_detect_raises_after_max_retries(
        self, _mock_sleep: MagicMock, _mock_handle: MagicMock
    ) -> None:
        with patch(f"{HF_SPACE_MODULE}.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.predict.side_effect = Exception("API 장애")
            mock_client_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="Detection API 호출 실패"):
                self.detector.detect("test.png")

        assert mock_client.predict.call_count == 4

    def test_invalid_schema_fails_immediately(self, _mock_handle: MagicMock) -> None:
        with patch(f"{HF_SPACE_MODULE}.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.predict.return_value = {"unexpected": "schema"}
            mock_client_cls.return_value = mock_client

            with pytest.raises(ValidationError):
                self.detector.detect("test.png")

        assert mock_client.predict.call_count == 1
