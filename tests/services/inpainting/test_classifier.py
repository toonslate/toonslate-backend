"""RegionClassifier 테스트"""

from src.schemas.pipeline import BBox, TextRegion
from src.services.inpainting.classifier import RegionClassifier

BUBBLE = BBox(x1=0, y1=0, x2=100, y2=100)
FAR_BUBBLE = BBox(x1=500, y1=500, x2=600, y2=600)


def _region(index: int, x1: float, y1: float, x2: float, y2: float) -> TextRegion:
    return TextRegion(index=index, text_bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2))


class TestRegionClassifier:
    def setup_method(self) -> None:
        self.classifier = RegionClassifier()

    def test_bubble_region_classified(self) -> None:
        region = _region(0, 10, 10, 90, 90)
        bubble_regions, free_regions = self.classifier.classify([region], [BUBBLE])
        assert len(bubble_regions) == 1
        assert len(free_regions) == 0
        assert bubble_regions[0].bubble_bbox == BUBBLE

    def test_free_region_classified(self) -> None:
        region = _region(0, 10, 10, 90, 90)
        bubble_regions, free_regions = self.classifier.classify([region], [FAR_BUBBLE])
        assert len(bubble_regions) == 0
        assert len(free_regions) == 1
        assert free_regions[0].bubble_bbox is None

    def test_mixed_regions(self) -> None:
        inside = _region(0, 10, 10, 90, 90)
        outside = _region(1, 200, 200, 300, 300)
        bubble_regions, free_regions = self.classifier.classify([inside, outside], [BUBBLE])
        assert len(bubble_regions) == 1
        assert len(free_regions) == 1
        assert bubble_regions[0].index == 0
        assert free_regions[0].index == 1

    def test_empty_regions(self) -> None:
        bubble_regions, free_regions = self.classifier.classify([], [BUBBLE])
        assert bubble_regions == []
        assert free_regions == []

    def test_empty_bubbles_all_free(self) -> None:
        region = _region(0, 10, 10, 90, 90)
        bubble_regions, free_regions = self.classifier.classify([region], [])
        assert len(bubble_regions) == 0
        assert len(free_regions) == 1

    def test_does_not_mutate_original(self) -> None:
        region = _region(0, 10, 10, 90, 90)
        self.classifier.classify([region], [BUBBLE])
        assert region.bubble_bbox is None
