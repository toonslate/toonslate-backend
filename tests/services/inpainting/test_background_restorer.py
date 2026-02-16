"""IOPaintRestorer 테스트"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.schemas.pipeline import BBox, TextRegion
from src.services.inpainting.background_restorer import IOPaintRestorer

MODULE = "src.services.inpainting.background_restorer"


def _free_region(index: int, x1: float, y1: float, x2: float, y2: float) -> TextRegion:
    return TextRegion(index=index, text_bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2))


def _image(w: int = 200, h: int = 200) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


class TestIOPaintRestorer:
    def setup_method(self) -> None:
        self.restorer = IOPaintRestorer(space_url="http://test:7860")

    @patch(f"{MODULE}.httpx")
    def test_restore_returns_image_and_regions(self, mock_httpx: MagicMock) -> None:
        result_image = _image()
        mock_resp = MagicMock()
        mock_resp.content = self._fake_png(result_image)
        mock_httpx.Client.return_value.__enter__.return_value.post.return_value = mock_resp

        image = _image()
        region = _free_region(0, 10, 10, 90, 90)
        clean, regions = self.restorer.restore(image, [region])

        assert clean.shape[0] > 0
        assert len(regions) == 1
        assert regions[0].inpaint_bbox is not None
        assert regions[0].render_bbox is not None

    @patch(f"{MODULE}.httpx")
    def test_restore_empty_regions(self, mock_httpx: MagicMock) -> None:
        image = _image()
        clean, regions = self.restorer.restore(image, [])
        assert np.array_equal(clean, image)
        assert regions == []

    @patch(f"{MODULE}.httpx")
    def test_restore_skips_out_of_bounds_regions(self, mock_httpx: MagicMock) -> None:
        image = _image()
        region = _free_region(0, 300, 300, 400, 400)
        clean, regions = self.restorer.restore(image, [region])

        assert np.array_equal(clean, image)
        assert regions == []
        mock_httpx.Client.return_value.__enter__.return_value.post.assert_not_called()

    @patch(f"{MODULE}.httpx")
    def test_api_timeout_raises_inpainting_error(self, mock_httpx: MagicMock) -> None:
        import httpx

        from src.services.inpainting.solid_fill import InpaintingError

        mock_httpx.Client.return_value.__enter__.return_value.post.side_effect = (
            httpx.TimeoutException("timeout")
        )
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError

        image = _image()
        region = _free_region(0, 10, 10, 90, 90)
        with pytest.raises(InpaintingError, match="타임아웃"):
            self.restorer.restore(image, [region])

    @patch(f"{MODULE}.httpx")
    def test_api_http_error_raises_inpainting_error(self, mock_httpx: MagicMock) -> None:
        import httpx

        from src.services.inpainting.solid_fill import InpaintingError

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_httpx.Client.return_value.__enter__.return_value.post.side_effect = (
            httpx.HTTPStatusError("error", request=MagicMock(), response=mock_resp)
        )
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError

        image = _image()
        region = _free_region(0, 10, 10, 90, 90)
        with pytest.raises(InpaintingError, match="API 오류"):
            self.restorer.restore(image, [region])

    @patch(f"{MODULE}.httpx")
    def test_restore_mask_delegates_to_api(self, mock_httpx: MagicMock) -> None:
        result_image = _image(100, 100)
        mock_resp = MagicMock()
        mock_resp.content = self._fake_png(result_image)
        mock_httpx.Client.return_value.__enter__.return_value.post.return_value = mock_resp

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.full((100, 100), 255, dtype=np.uint8)
        result = self.restorer.restore_mask(image, mask)

        assert result.ndim == 3
        mock_httpx.Client.return_value.__enter__.return_value.post.assert_called_once()

    def _fake_png(self, image: np.ndarray) -> bytes:
        from io import BytesIO

        from PIL import Image

        pil = Image.fromarray(image)
        buf = BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()
