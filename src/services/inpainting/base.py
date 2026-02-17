"""Inpainting Protocol 정의

스왑 가능한 인페인팅 구현을 위한 인터페이스 정의
"""

from typing import Protocol

import numpy as np

from src.schemas.pipeline import BBox, TextRegion


class BubbleCleaner(Protocol):
    """말풍선 텍스트 제거기

    단색 배경(말풍선 내부) 텍스트를 정리.
    regions에는 bubble_bbox가 이미 설정된 상태.
    """

    def clean(
        self, image: np.ndarray, regions: list[TextRegion]
    ) -> tuple[np.ndarray, list[TextRegion]]:
        """말풍선 영역 텍스트 제거

        Returns:
            (처리된 이미지, inpaint_bbox/render_bbox가 설정된 regions)
        """
        ...


class BackgroundRestorer(Protocol):
    """복잡한 배경의 텍스트 영역을 AI로 복원"""

    def restore(
        self, image: np.ndarray, regions: list[TextRegion]
    ) -> tuple[np.ndarray, list[TextRegion]]:
        """free text 영역 텍스트 제거 (마스크 생성 → AI 복원)

        Returns:
            (복원된 이미지, inpaint_bbox/render_bbox가 설정된 regions)
        """
        ...

    def restore_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """마스크 기반 복원 (Erase API용)

        Args:
            image: RGB 이미지
            mask: 그레이스케일 마스크 (255 = 제거 영역)

        Returns:
            복원된 RGB 이미지
        """
        ...


class Inpainter(Protocol):
    """인페인팅 인터페이스

    구현체:
    - RoutedInpainting: 영역별 라우팅 (BubbleCleaner + BackgroundRestorer)
    - SolidFillInpainting: 단색 채우기 (레거시)
    - ReplicateLamaInpainting: Replicate LaMa API (레거시)
    - IOPaintLamaInpainting: IOPaint HuggingFace Space (레거시)
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
            TextRegion에 bubble_bbox, inpaint_bbox, render_bbox가 채워짐
        """
        ...

    def inpaint_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """마스크 기반 inpainting (erase 서비스용)

        Args:
            image: RGB 이미지 (numpy 배열)
            mask: 그레이스케일 마스크 (255 = 제거 영역)

        Returns:
            inpainting된 RGB 이미지
        """
        ...
