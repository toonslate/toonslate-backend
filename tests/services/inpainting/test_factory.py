"""Inpainting 팩토리 테스트"""

from unittest.mock import patch

from src.services.inpainting import get_inpainting, set_inpainting
from src.services.inpainting.inpainter import RoutedInpainting
from src.services.inpainting.solid_fill import SolidFillInpainting


class TestGetInpainting:
    def setup_method(self) -> None:
        set_inpainting(None)

    def test_default_returns_solid_fill(self) -> None:
        with patch("src.services.inpainting.get_settings") as mock_settings:
            mock_settings.return_value.inpainting_provider = "solid_fill"
            inpainter = get_inpainting()
        assert isinstance(inpainter, SolidFillInpainting)

    def test_iopaint_lama_returns_routed(self) -> None:
        with patch("src.services.inpainting.get_settings") as mock_settings:
            mock_settings.return_value.inpainting_provider = "iopaint_lama"
            mock_settings.return_value.iopaint_space_url = "http://test:7860"
            mock_settings.return_value.iopaint_timeout = 60
            inpainter = get_inpainting()
        assert isinstance(inpainter, RoutedInpainting)

    def test_solid_fill_returns_legacy(self) -> None:
        with patch("src.services.inpainting.get_settings") as mock_settings:
            mock_settings.return_value.inpainting_provider = "solid_fill"
            inpainter = get_inpainting()
        assert isinstance(inpainter, SolidFillInpainting)

    def test_set_inpainting_overrides(self) -> None:
        mock = MockInpainter()
        set_inpainting(mock)
        assert get_inpainting() is mock

    def test_set_inpainting_none_resets(self) -> None:
        mock = MockInpainter()
        set_inpainting(mock)
        set_inpainting(None)
        with patch("src.services.inpainting.get_settings") as mock_settings:
            mock_settings.return_value.inpainting_provider = "solid_fill"
            inpainter = get_inpainting()
        assert isinstance(inpainter, SolidFillInpainting)

    def test_get_inpainting_caches(self) -> None:
        with patch("src.services.inpainting.get_settings") as mock_settings:
            mock_settings.return_value.inpainting_provider = "solid_fill"
            first = get_inpainting()
            second = get_inpainting()
        assert first is second


class MockInpainter:
    def inpaint(self, image: object, text_regions: object, bubble_bboxes: object) -> object:
        return None

    def inpaint_mask(self, image: object, mask: object) -> object:
        return None
