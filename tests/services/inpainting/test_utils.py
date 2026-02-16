"""Inpainting utils 순수 함수 테스트"""

import pytest

from src.schemas.pipeline import BBox
from src.services.inpainting.utils import (
    INSCRIBED_RATIO,
    calc_overlap_ratio,
    calc_render_bbox,
    clip_to_bounds,
    find_bubble,
    inscribed_rect,
)


class TestCalcOverlapRatio:
    def test_complete_overlap(self) -> None:
        box = BBox(x1=0, y1=0, x2=100, y2=100)
        assert calc_overlap_ratio(box, box) == 1.0

    def test_box_a_inside_box_b(self) -> None:
        box_a = BBox(x1=10, y1=10, x2=50, y2=50)
        box_b = BBox(x1=0, y1=0, x2=100, y2=100)
        assert calc_overlap_ratio(box_a, box_b) == 1.0

    def test_partial_overlap(self) -> None:
        box_a = BBox(x1=0, y1=0, x2=100, y2=100)
        box_b = BBox(x1=50, y1=0, x2=150, y2=100)
        assert calc_overlap_ratio(box_a, box_b) == pytest.approx(0.5)

    def test_no_overlap(self) -> None:
        box_a = BBox(x1=0, y1=0, x2=50, y2=50)
        box_b = BBox(x1=100, y1=100, x2=200, y2=200)
        assert calc_overlap_ratio(box_a, box_b) == 0.0

    def test_zero_area_box_a(self) -> None:
        box_a = BBox(x1=10, y1=10, x2=10, y2=50)
        box_b = BBox(x1=0, y1=0, x2=100, y2=100)
        assert calc_overlap_ratio(box_a, box_b) == 0.0


class TestClipToBounds:
    def test_within_bounds(self) -> None:
        bbox = BBox(x1=10, y1=20, x2=90, y2=80)
        result = clip_to_bounds(bbox, 100, 100)
        assert result == bbox

    def test_clips_to_image_bounds(self) -> None:
        bbox = BBox(x1=-10, y1=-20, x2=150, y2=200)
        result = clip_to_bounds(bbox, 100, 100)
        assert result == BBox(x1=0, y1=0, x2=100, y2=100)

    def test_completely_outside_returns_zero_area(self) -> None:
        bbox = BBox(x1=300, y1=300, x2=400, y2=400)
        result = clip_to_bounds(bbox, 200, 200)
        assert result.width == 0
        assert result.height == 0


class TestInscribedRect:
    def test_default_ratio(self) -> None:
        bubble = BBox(x1=0, y1=0, x2=100, y2=100)
        result = inscribed_rect(bubble)
        cx, cy = 50.0, 50.0
        half = 50.0 * INSCRIBED_RATIO
        assert result.x1 == pytest.approx(cx - half)
        assert result.y1 == pytest.approx(cy - half)
        assert result.x2 == pytest.approx(cx + half)
        assert result.y2 == pytest.approx(cy + half)

    def test_custom_ratio(self) -> None:
        bubble = BBox(x1=0, y1=0, x2=200, y2=100)
        result = inscribed_rect(bubble, ratio=0.5)
        assert result.x1 == pytest.approx(50.0)
        assert result.y1 == pytest.approx(25.0)
        assert result.x2 == pytest.approx(150.0)
        assert result.y2 == pytest.approx(75.0)


class TestFindBubble:
    def test_match_above_threshold(self) -> None:
        text = BBox(x1=10, y1=10, x2=90, y2=90)
        bubble = BBox(x1=0, y1=0, x2=100, y2=100)
        result = find_bubble(text, [bubble])
        assert result == bubble

    def test_no_match_below_threshold(self) -> None:
        text = BBox(x1=0, y1=0, x2=100, y2=100)
        bubble = BBox(x1=80, y1=80, x2=200, y2=200)
        result = find_bubble(text, [bubble])
        assert result is None

    def test_empty_bubbles(self) -> None:
        text = BBox(x1=0, y1=0, x2=100, y2=100)
        assert find_bubble(text, []) is None

    def test_picks_best_overlap(self) -> None:
        text = BBox(x1=10, y1=10, x2=90, y2=90)
        small_bubble = BBox(x1=0, y1=0, x2=60, y2=60)
        big_bubble = BBox(x1=0, y1=0, x2=100, y2=100)
        result = find_bubble(text, [small_bubble, big_bubble])
        assert result == big_bubble


class TestCalcRenderBbox:
    def test_with_bubble_returns_inscribed(self) -> None:
        bubble = BBox(x1=0, y1=0, x2=100, y2=100)
        inpaint = BBox(x1=10, y1=10, x2=90, y2=90)
        result = calc_render_bbox(bubble, inpaint)
        expected = inscribed_rect(bubble, INSCRIBED_RATIO)
        assert result == expected

    def test_without_bubble_returns_inpaint_bbox(self) -> None:
        inpaint = BBox(x1=10, y1=10, x2=90, y2=90)
        result = calc_render_bbox(None, inpaint)
        assert result == inpaint
