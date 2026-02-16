"""SolidFillBubbleCleaner 테스트"""

import numpy as np

from src.schemas.pipeline import BBox, TextRegion
from src.services.inpainting.bubble_cleaner import SolidFillBubbleCleaner


def _bubble_region(index: int, text_bbox: BBox, bubble_bbox: BBox) -> TextRegion:
    return TextRegion(index=index, text_bbox=text_bbox, bubble_bbox=bubble_bbox)


def _white_image(w: int = 200, h: int = 200) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


BUBBLE = BBox(x1=20, y1=20, x2=180, y2=180)
TEXT = BBox(x1=50, y1=50, x2=150, y2=150)


class TestSolidFillBubbleCleaner:
    def setup_method(self) -> None:
        self.cleaner = SolidFillBubbleCleaner()

    def test_clean_returns_image_and_regions(self) -> None:
        image = _white_image()
        region = _bubble_region(0, TEXT, BUBBLE)
        result_image, result_regions = self.cleaner.clean(image, [region])

        assert result_image.shape == image.shape
        assert len(result_regions) == 1

    def test_sets_inpaint_and_render_bbox(self) -> None:
        image = _white_image()
        region = _bubble_region(0, TEXT, BUBBLE)
        _, result_regions = self.cleaner.clean(image, [region])

        assert result_regions[0].inpaint_bbox is not None
        assert result_regions[0].render_bbox is not None
        assert result_regions[0].bubble_bbox == BUBBLE

    def test_empty_regions(self) -> None:
        image = _white_image()
        result_image, result_regions = self.cleaner.clean(image, [])
        assert np.array_equal(result_image, image)
        assert result_regions == []

    def test_does_not_mutate_original_image(self) -> None:
        image = _white_image()
        original = image.copy()
        region = _bubble_region(0, TEXT, BUBBLE)
        self.cleaner.clean(image, [region])
        assert np.array_equal(image, original)

    def test_inpaint_bbox_within_bubble(self) -> None:
        image = _white_image()
        region = _bubble_region(0, TEXT, BUBBLE)
        _, result_regions = self.cleaner.clean(image, [region])

        inpaint = result_regions[0].inpaint_bbox
        assert inpaint is not None
        assert inpaint.x1 >= BUBBLE.x1
        assert inpaint.y1 >= BUBBLE.y1
        assert inpaint.x2 <= BUBBLE.x2
        assert inpaint.y2 <= BUBBLE.y2
