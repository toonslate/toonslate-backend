"""Inpainting 공통 유틸리티 함수"""

import time
from pathlib import Path

import cv2
import numpy as np

from src.schemas.pipeline import BBox, TextRegion

INSCRIBED_RATIO = 0.65  # 타원 내접 직사각형 비율 (수학적 최대: 0.707)
OVERLAP_THRESHOLD = 0.5  # bubble 매칭 최소 겹침 비율


def calc_overlap_ratio(box_a: BBox, box_b: BBox) -> float:
    """두 박스의 겹침 비율 계산 (box_a 면적 기준)"""
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
    """박스를 이미지 경계 [0, width] x [0, height] 내로 클리핑

    완전히 경계 밖이면 zero-area BBox 반환.
    """
    return BBox(
        x1=min(width, max(0, bbox.x1)),
        y1=min(height, max(0, bbox.y1)),
        x2=min(width, max(0, bbox.x2)),
        y2=min(height, max(0, bbox.y2)),
    )


def inscribed_rect(bubble: BBox, ratio: float = INSCRIBED_RATIO) -> BBox:
    """타원에 내접하는 직사각형 (ratio=0.707이 수학적 최대)"""
    cx, cy = bubble.center
    hw, hh = bubble.width / 2, bubble.height / 2
    return BBox(x1=cx - hw * ratio, y1=cy - hh * ratio, x2=cx + hw * ratio, y2=cy + hh * ratio)


def find_bubble(text_bbox: BBox, bubbles: list[BBox]) -> BBox | None:
    """텍스트와 가장 많이 겹치는 bubble 반환 (threshold 이상만)"""
    best, best_overlap = None, 0.0

    for bubble in bubbles:
        overlap = calc_overlap_ratio(text_bbox, bubble)
        if overlap > best_overlap:
            best, best_overlap = bubble, overlap

    return best if best_overlap > OVERLAP_THRESHOLD else None


def calc_render_bbox(bubble: BBox | None, inpaint_bbox: BBox) -> BBox:
    """렌더링용 안전 영역 계산"""
    if bubble:
        return inscribed_rect(bubble, INSCRIBED_RATIO)
    return inpaint_bbox


def create_mask(shape: tuple[int, int], regions: list[TextRegion]) -> np.ndarray:
    """검정 배경에 text_bbox를 흰색으로 채운 마스크 생성"""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for region in regions:
        x1, y1, x2, y2 = region.text_bbox.to_tuple()
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return mask


def save_debug_images(
    debug_dir: Path,
    image: np.ndarray,
    mask: np.ndarray,
    regions: list[TextRegion],
) -> None:
    """디버그용 마스크/오버레이 이미지 저장"""
    timestamp = int(time.time() * 1000)

    cv2.imwrite(str(debug_dir / f"{timestamp}_1_mask.png"), mask)

    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

    for i, region in enumerate(regions):
        tx1, ty1, tx2, ty2 = region.text_bbox.to_tuple()
        cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
        cv2.putText(overlay, f"T{i}", (tx1, ty1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if region.inpaint_bbox:
            ix1, iy1, ix2, iy2 = region.inpaint_bbox.to_tuple()
            cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2), (255, 0, 0), 2)

        if region.bubble_bbox:
            bx1, by1, bx2, by2 = region.bubble_bbox.to_tuple()
            cv2.ellipse(
                overlay,
                ((bx1 + bx2) // 2, (by1 + by2) // 2),
                ((bx2 - bx1) // 2, (by2 - by1) // 2),
                0,
                0,
                360,
                (255, 255, 0),
                2,
            )

        if region.render_bbox:
            rx1, ry1, rx2, ry2 = region.render_bbox.to_tuple()
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)

    cv2.imwrite(str(debug_dir / f"{timestamp}_2_overlay.png"), overlay)


def convert_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    """RGB/RGBA/Grayscale 이미지를 BGR로 변환"""
    if len(img_rgb.shape) == 2:
        return cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2BGR)
    if img_rgb.shape[2] == 4:
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
