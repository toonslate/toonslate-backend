"""영역별 라우팅 기반 인페인팅"""

import numpy as np

from src.schemas.pipeline import BBox, TextRegion
from src.services.inpainting.base import BackgroundRestorer, BubbleCleaner
from src.services.inpainting.classifier import RegionClassifier


class RoutedInpainting:
    """텍스트 영역을 분류하여 적합한 백엔드로 위임

    - 말풍선 영역 → BubbleCleaner (단색 채움)
    - 자유 텍스트 → BackgroundRestorer (AI 복원)
    """

    def __init__(
        self,
        classifier: RegionClassifier,
        bubble_cleaner: BubbleCleaner,
        background_restorer: BackgroundRestorer,
    ) -> None:
        self._classifier = classifier
        self._bubble_cleaner = bubble_cleaner
        self._background_restorer = background_restorer

    def inpaint(
        self,
        image: np.ndarray,
        text_regions: list[TextRegion],
        bubble_bboxes: list[BBox],
    ) -> tuple[np.ndarray, list[TextRegion]]:
        bubble_regions, free_regions = self._classifier.classify(text_regions, bubble_bboxes)

        image, bubble_updated = self._bubble_cleaner.clean(image, bubble_regions)
        image, free_updated = self._background_restorer.restore(image, free_regions)

        all_regions = bubble_updated + free_updated
        all_regions.sort(key=lambda r: r.index)

        return image, all_regions

    def inpaint_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self._background_restorer.restore_mask(image, mask)
