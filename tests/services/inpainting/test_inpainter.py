"""RoutedInpainting 테스트"""

from unittest.mock import MagicMock

import numpy as np

from src.schemas.pipeline import BBox, TextRegion
from src.services.inpainting.classifier import RegionClassifier
from src.services.inpainting.inpainter import RoutedInpainting


def _region(index: int, x1: float, y1: float, x2: float, y2: float) -> TextRegion:
    return TextRegion(index=index, text_bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2))


def _image() -> np.ndarray:
    return np.full((200, 200, 3), 128, dtype=np.uint8)


BUBBLE = BBox(x1=0, y1=0, x2=100, y2=100)


class TestRoutedInpainting:
    def setup_method(self) -> None:
        self.bubble_cleaner = MagicMock()
        self.background_restorer = MagicMock()
        self.inpainter = RoutedInpainting(
            classifier=RegionClassifier(),
            bubble_cleaner=self.bubble_cleaner,
            background_restorer=self.background_restorer,
        )

    def test_delegates_bubble_regions_to_cleaner(self) -> None:
        image = _image()
        cleaned_image = _image()
        bubble_region = TextRegion(
            index=0,
            text_bbox=BBox(x1=10, y1=10, x2=90, y2=90),
            bubble_bbox=BUBBLE,
            inpaint_bbox=BBox(x1=10, y1=10, x2=90, y2=90),
            render_bbox=BBox(x1=10, y1=10, x2=90, y2=90),
        )
        self.bubble_cleaner.clean.return_value = (cleaned_image, [bubble_region])
        self.background_restorer.restore.return_value = (cleaned_image, [])

        region = _region(0, 10, 10, 90, 90)
        result_image, result_regions = self.inpainter.inpaint(image, [region], [BUBBLE])

        self.bubble_cleaner.clean.assert_called_once()
        assert len(result_regions) == 1

    def test_delegates_free_regions_to_restorer(self) -> None:
        image = _image()
        cleaned_image = _image()
        free_region = TextRegion(
            index=0,
            text_bbox=BBox(x1=200, y1=200, x2=300, y2=300),
            inpaint_bbox=BBox(x1=200, y1=200, x2=300, y2=300),
            render_bbox=BBox(x1=200, y1=200, x2=300, y2=300),
        )
        self.bubble_cleaner.clean.return_value = (image, [])
        self.background_restorer.restore.return_value = (cleaned_image, [free_region])

        region = _region(0, 200, 200, 300, 300)
        result_image, result_regions = self.inpainter.inpaint(image, [region], [BUBBLE])

        self.background_restorer.restore.assert_called_once()
        assert len(result_regions) == 1

    def test_mixed_regions_sorted_by_index(self) -> None:
        image = _image()
        bubble_result = TextRegion(
            index=0,
            text_bbox=BBox(x1=10, y1=10, x2=90, y2=90),
            bubble_bbox=BUBBLE,
            inpaint_bbox=BBox(x1=10, y1=10, x2=90, y2=90),
            render_bbox=BBox(x1=10, y1=10, x2=90, y2=90),
        )
        free_result = TextRegion(
            index=1,
            text_bbox=BBox(x1=200, y1=200, x2=300, y2=300),
            inpaint_bbox=BBox(x1=200, y1=200, x2=300, y2=300),
            render_bbox=BBox(x1=200, y1=200, x2=300, y2=300),
        )
        self.bubble_cleaner.clean.return_value = (image, [bubble_result])
        self.background_restorer.restore.return_value = (image, [free_result])

        inside = _region(0, 10, 10, 90, 90)
        outside = _region(1, 200, 200, 300, 300)
        _, result_regions = self.inpainter.inpaint(image, [inside, outside], [BUBBLE])

        assert len(result_regions) == 2
        assert result_regions[0].index == 0
        assert result_regions[1].index == 1

    def test_empty_regions(self) -> None:
        image = _image()
        self.bubble_cleaner.clean.return_value = (image, [])
        self.background_restorer.restore.return_value = (image, [])

        result_image, result_regions = self.inpainter.inpaint(image, [], [])
        assert result_regions == []

    def test_inpaint_mask_delegates_to_restorer(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.full((100, 100), 255, dtype=np.uint8)
        expected = np.ones((100, 100, 3), dtype=np.uint8)
        self.background_restorer.restore_mask.return_value = expected

        result = self.inpainter.inpaint_mask(image, mask)

        self.background_restorer.restore_mask.assert_called_once_with(image, mask)
        assert np.array_equal(result, expected)
