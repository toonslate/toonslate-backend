"""Inpainting 백엔드 Protocol

스왑 가능한 인페인팅 구현을 위한 인터페이스 정의
"""

from typing import Protocol

import numpy as np

from src.schemas.pipeline import BBox, TextRegion


class InpaintingBackend(Protocol):
    """인페인팅 백엔드 인터페이스

    구현체:
    - SolidFillInpainting: 단색 채우기 (MVP)
    - ReplicateLamaInpainting: Replicate LaMa API (향후)
    """

    def inpaint(
        self,
        image: np.ndarray,
        text_regions: list[TextRegion],
        bubble_bboxes: list[BBox],
    ) -> tuple[np.ndarray, list[TextRegion]]:
        """텍스트 영역을 지우고 깨끗한 이미지 반환

        Args:
            image: BGR 이미지 (cv2 형식)
            text_regions: 텍스트 영역 리스트 (text_bbox 필수)
            bubble_bboxes: 말풍선 바운딩 박스 리스트 (bubble 매칭용)

        Returns:
            (처리된 이미지, 업데이트된 TextRegion 리스트)
            TextRegion에 bubble_bbox, fill_bbox, render_bbox가 채워짐
        """
        ...
