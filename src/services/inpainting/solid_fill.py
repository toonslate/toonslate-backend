"""단색 채우기 Inpainting (MVP)"""

import cv2
import numpy as np

from src.schemas.pipeline import BBox, TextRegion

INSCRIBED_RATIO = 0.65  # 타원 내접 직사각형 비율 (수학적 최대: 0.707)
OVERLAP_THRESHOLD = 0.5  # bubble 매칭 최소 겹침 비율


class InpaintingError(Exception):
    pass


def calc_overlap_ratio(box_a: BBox, box_b: BBox) -> float:
    """두 박스의 겹침 비율 계산 (box_a 면적 기준, 순수 함수)"""
    area_a = box_a.width * box_a.height
    if area_a <= 0:
        return 0.0

    ix1 = max(box_a.x1, box_b.x1)
    iy1 = max(box_a.y1, box_b.y1)
    ix2 = min(box_a.x2, box_b.x2)
    iy2 = min(box_a.y2, box_b.y2)

    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    return (ix2 - ix1) * (iy2 - iy1) / area_a


def clip_to_bounds(bbox: BBox, width: int, height: int) -> BBox:
    """박스를 이미지 경계 내로 클리핑 (순수 함수)"""
    return BBox(
        x1=max(0, bbox.x1),
        y1=max(0, bbox.y1),
        x2=min(width, bbox.x2),
        y2=min(height, bbox.y2),
    )


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
            bubble = self._find_bubble(region.text_bbox, bubble_bboxes)
            fill_bbox = self._calc_fill_bbox(region.text_bbox, bubble, (w, h))
            render_bbox = self._calc_render_bbox(bubble, fill_bbox)

            color = self._extract_bg_color(image, fill_bbox)
            x1, y1, x2, y2 = fill_bbox.to_tuple()
            cv2.rectangle(result, (x1, y1), (x2, y2), color, -1)

            updated_regions.append(
                TextRegion(
                    index=region.index,
                    text_bbox=region.text_bbox,
                    bubble_bbox=bubble,
                    fill_bbox=fill_bbox,
                    render_bbox=render_bbox,
                )
            )

        return result, updated_regions

    def _find_bubble(self, text_bbox: BBox, bubbles: list[BBox]) -> BBox | None:
        """텍스트와 가장 많이 겹치는 bubble 반환 (threshold 이상만)"""
        best, best_overlap = None, 0.0

        for bubble in bubbles:
            overlap = calc_overlap_ratio(text_bbox, bubble)
            if overlap > best_overlap:
                best, best_overlap = bubble, overlap

        return best if best_overlap > OVERLAP_THRESHOLD else None

    def _inscribed_rect(self, bubble: BBox, ratio: float) -> BBox:
        """타원에 내접하는 직사각형 (ratio=0.707이 수학적 최대)"""
        cx, cy = bubble.center
        hw, hh = bubble.width / 2, bubble.height / 2
        return BBox(x1=cx - hw * ratio, y1=cy - hh * ratio, x2=cx + hw * ratio, y2=cy + hh * ratio)

    def _calc_fill_bbox(self, text: BBox, bubble: BBox | None, img_size: tuple[int, int]) -> BBox:
        w, h = img_size

        if bubble:
            inscribed = self._inscribed_rect(bubble, INSCRIBED_RATIO)
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

    def _calc_render_bbox(self, bubble: BBox | None, fill_bbox: BBox) -> BBox:
        if bubble:
            return self._inscribed_rect(bubble, INSCRIBED_RATIO)
        return fill_bbox

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
