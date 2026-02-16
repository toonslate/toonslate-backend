"""Detection 팩토리 테스트"""

from unittest.mock import patch

import pytest

from src.services.detection import get_detection, set_detection
from src.services.detection.hf_space import HFSpaceDetection


# TODO: 캐싱 테스트 추가 (get_detection() 두 번 호출 시 같은 인스턴스 반환)
class TestGetDetection:
    def setup_method(self) -> None:
        set_detection(None)

    def test_default_returns_hf_space(self) -> None:
        backend = get_detection()
        assert isinstance(backend, HFSpaceDetection)

    def test_set_detection_overrides_factory(self) -> None:
        mock = MockDetector()
        set_detection(mock)
        assert get_detection() is mock

    def test_set_detection_none_resets(self) -> None:
        mock = MockDetector()
        set_detection(mock)
        set_detection(None)

        backend = get_detection()
        assert isinstance(backend, HFSpaceDetection)

    def test_unknown_provider_raises(self) -> None:
        with patch("src.services.detection.get_settings") as mock_settings:
            mock_settings.return_value.detection_provider = "unknown"
            with pytest.raises(ValueError, match="Unknown detection provider"):
                get_detection()


class MockDetector:
    def detect(self, image_path: str) -> object:
        return None
