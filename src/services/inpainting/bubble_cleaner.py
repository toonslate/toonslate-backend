"""SolidFill 기반 말풍선 텍스트 제거"""

import cv2
import numpy as np

from src.schemas.pipeline import BBox, TextRegion
from src.services.inpainting.utils import (
    INSCRIBED_RATIO,
    calc_render_bbox,
    clip_to_bounds,
    inscribed_rect,
)


class SolidFillBubbleCleaner:
    """말풍선 내부 텍스트를 배경색 추출 후 단색으로 채움"""

    def __init__(self, padding_ratio: float = 0.2):
        self._padding_ratio = padding_ratio

    def clean(
        self, image: np.ndarray, regions: list[TextRegion]
    ) -> tuple[np.ndarray, list[TextRegion]]:
        h, w = image.shape[:2]
        result = image.copy()
        updated: list[TextRegion] = []

        for region in regions:
            bubble = region.bubble_bbox
            if bubble is None:
                continue

            inpaint_bbox = self._calc_inpaint_bbox(region.text_bbox, bubble, (w, h))
            render_bbox = calc_render_bbox(bubble, inpaint_bbox)

            color = self._extract_bg_color(image, inpaint_bbox)
            x1, y1, x2, y2 = inpaint_bbox.to_tuple()
            cv2.rectangle(result, (x1, y1), (x2, y2), color, -1)

            updated.append(
                TextRegion(
                    index=region.index,
                    text_bbox=region.text_bbox,
                    bubble_bbox=bubble,
                    inpaint_bbox=inpaint_bbox,
                    render_bbox=render_bbox,
                )
            )

        return result, updated

    def _calc_inpaint_bbox(self, text: BBox, bubble: BBox, img_size: tuple[int, int]) -> BBox:
        # TODO: text가 inscribed rect 바깥에 있으면 clamp 후 x1>x2 역전 가능
        #       BBox 자동 정렬로 말풍선 밖까지 확장될 수 있음. 교집합 계산으로 개선 필요.
        w, h = img_size
        inscribed = inscribed_rect(bubble, INSCRIBED_RATIO)
        pad_x = text.width * self._padding_ratio
        pad_y = text.height * self._padding_ratio
        bbox = BBox(
            x1=max(text.x1 - pad_x, inscribed.x1),
            y1=max(text.y1 - pad_y, inscribed.y1),
            x2=min(text.x2 + pad_x, inscribed.x2),
            y2=min(text.y2 + pad_y, inscribed.y2),
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
