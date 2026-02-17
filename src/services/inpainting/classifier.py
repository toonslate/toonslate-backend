"""텍스트 영역 분류기: 말풍선 vs 자유 텍스트"""

from src.schemas.pipeline import BBox, TextRegion
from src.services.inpainting.utils import find_bubble


class RegionClassifier:
    """텍스트 영역을 말풍선/자유 텍스트로 분류"""

    def classify(
        self, text_regions: list[TextRegion], bubble_bboxes: list[BBox]
    ) -> tuple[list[TextRegion], list[TextRegion]]:
        """(bubble_regions, free_regions) 반환

        bubble_regions에는 bubble_bbox가 설정된 새 TextRegion 생성.
        원본 text_regions는 변이하지 않음.
        """
        bubble_regions: list[TextRegion] = []
        free_regions: list[TextRegion] = []

        for region in text_regions:
            bubble = find_bubble(region.text_bbox, bubble_bboxes)
            if bubble:
                bubble_regions.append(
                    TextRegion(
                        index=region.index,
                        text_bbox=region.text_bbox,
                        bubble_bbox=bubble,
                    )
                )
            else:
                free_regions.append(
                    TextRegion(
                        index=region.index,
                        text_bbox=region.text_bbox,
                    )
                )

        return bubble_regions, free_regions
