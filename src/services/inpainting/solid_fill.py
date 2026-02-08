"""단색 채우기 Inpainting (MVP)"""

import cv2
import numpy as np

from src.schemas.pipeline import BBox, TextRegion
from src.services.inpainting.utils import (
    INSCRIBED_RATIO,
    calc_render_bbox,
    clip_to_bounds,
    find_bubble,
    inscribed_rect,
)


class InpaintingError(Exception):
    pass


class SolidFillInpainting:
    """가장자리 픽셀에서 배경색을 추출하여 영역을 단색으로 채움"""

    def __init__(self, padding_ratio: float = 0.2):
        self.padding_ratio = padding_ratio

    def inpaint(
        self,
        image: np.ndarray,
        text_regions: list[TextRegion],
        bubble_bboxes: list[BBox],
    ) -> tuple[np.ndarray, list[TextRegion]]:
        if image.size == 0:
            raise InpaintingError("유효하지 않은 이미지입니다")

        h, w = image.shape[:2]
        result = image.copy()
        updated_regions: list[TextRegion] = []

        for region in text_regions:
            bubble = find_bubble(region.text_bbox, bubble_bboxes)
            inpaint_bbox = self._calc_inpaint_bbox(region.text_bbox, bubble, (w, h))
            render_bbox = calc_render_bbox(bubble, inpaint_bbox)

            color = self._extract_bg_color(image, inpaint_bbox)
            x1, y1, x2, y2 = inpaint_bbox.to_tuple()
            cv2.rectangle(result, (x1, y1), (x2, y2), color, -1)

            updated_regions.append(
                TextRegion(
                    index=region.index,
                    text_bbox=region.text_bbox,
                    bubble_bbox=bubble,
                    inpaint_bbox=inpaint_bbox,
                    render_bbox=render_bbox,
                )
            )

        return result, updated_regions

    def inpaint_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """마스크 기반 inpainting (OpenCV TELEA 알고리즘)

        Args:
            image: RGB 이미지
            mask: 그레이스케일 마스크 (255 = 제거 영역)

        Returns:
            inpainting된 RGB 이미지
        """
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result_bgr = cv2.inpaint(image_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    def _calc_inpaint_bbox(
        self, text: BBox, bubble: BBox | None, img_size: tuple[int, int]
    ) -> BBox:
        w, h = img_size

        if bubble:
            inscribed = inscribed_rect(bubble, INSCRIBED_RATIO)
            pad_x, pad_y = text.width * self.padding_ratio, text.height * self.padding_ratio
            bbox = BBox(
                x1=max(text.x1 - pad_x, inscribed.x1),
                y1=max(text.y1 - pad_y, inscribed.y1),
                x2=min(text.x2 + pad_x, inscribed.x2),
                y2=min(text.y2 + pad_y, inscribed.y2),
            )
        else:
            pad_x, pad_y = text.width * 1.0, text.height * 0.3
            bbox = BBox(
                x1=text.x1 - pad_x,
                y1=text.y1 - pad_y,
                x2=text.x2 + pad_x,
                y2=text.y2 + pad_y,
            )

        return clip_to_bounds(bbox, w, h)

    def _extract_bg_color(self, image: np.ndarray, bbox: BBox) -> tuple[int, int, int]:
        x1, y1, x2, y2 = bbox.to_tuple()
        region = image[y1:y2, x1:x2]

        if region.size == 0:
            return (255, 255, 255)

        h, w = region.shape[:2]
        border = min(5, h // 4, w // 4)
        if border < 1:
            return (255, 255, 255)

        edges = np.vstack(
            [
                region[:border, :].reshape(-1, 3),
                region[-border:, :].reshape(-1, 3),
                region[:, :border].reshape(-1, 3),
                region[:, -border:].reshape(-1, 3),
            ]
        )

        brightness = np.mean(edges, axis=1)
        bright = edges[brightness > 180]
        color = np.median(bright if len(bright) > 10 else edges, axis=0)

        return (int(color[0]), int(color[1]), int(color[2]))
