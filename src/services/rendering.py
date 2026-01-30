"""텍스트 렌더링 서비스"""

import logging
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.schemas.pipeline import TextRegion, TranslationResult

logger = logging.getLogger(__name__)

FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


class RenderingError(Exception):
    pass


@lru_cache(maxsize=32)
def _get_font(size: int) -> ImageFont.FreeTypeFont:
    for font_path in FONT_PATHS:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    return cast(ImageFont.FreeTypeFont, ImageFont.load_default())


def _fit_text(
    text: str, width: int, height: int, draw: ImageDraw.ImageDraw
) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """박스에 맞는 최적 폰트와 줄바꿈된 텍스트 반환"""
    max_size = min(height // 2, 40)
    min_size = 8

    for size in range(max_size, min_size - 1, -1):
        font = _get_font(size)
        lines = _wrap_text(text, width, font, draw)

        if _text_fits(lines, width, height, size, draw, font):
            return font, lines

    return _get_font(min_size), _force_wrap(text, width, min_size, draw)


def _calc_chars_per_line(box_width: int, avg_char_width: float) -> int:
    """박스 너비와 평균 글자 너비로 한 줄 글자 수 계산 (순수 함수)"""
    return max(1, int((box_width * 0.8) / avg_char_width))


def _wrap_text(
    text: str, width: int, font: ImageFont.FreeTypeFont, draw: ImageDraw.ImageDraw
) -> list[str]:
    sample = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    bbox = draw.textbbox((0, 0), sample, font=font)
    avg_char_width = (bbox[2] - bbox[0]) / len(sample)
    chars_per_line = _calc_chars_per_line(width, avg_char_width)
    return textwrap.fill(text, width=chars_per_line).split("\n")


def _fits_in_box(text_width: float, text_height: float, box_width: int, box_height: int) -> bool:
    """텍스트가 박스 안에 들어가는지 확인 (순수 함수, 테스트 용이)"""
    return text_height <= box_height * 0.95 and text_width <= box_width * 0.95


def _measure_text_block(
    lines: list[str], font_size: int, draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont
) -> tuple[float, float]:
    """텍스트 블록의 너비와 높이 측정"""
    total_height = len(lines) * font_size * 1.3
    max_width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
    return max_width, total_height


def _text_fits(
    lines: list[str],
    width: int,
    height: int,
    font_size: int,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
) -> bool:
    text_width, text_height = _measure_text_block(lines, font_size, draw, font)
    return _fits_in_box(text_width, text_height, width, height)


def _force_wrap(text: str, width: int, font_size: int, draw: ImageDraw.ImageDraw) -> list[str]:
    """글자 단위 강제 줄바꿈 (fallback)"""
    font = _get_font(font_size)
    lines: list[str] = []
    current = ""

    for char in text:
        test = current + char
        if draw.textbbox((0, 0), test, font=font)[2] > width * 0.9:
            if current:
                lines.append(current)
            current = char
        else:
            current = test

    if current:
        lines.append(current)

    return lines


def _render_text_in_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> None:
    width, height = x2 - x1, y2 - y1
    if width < 10 or height < 10:
        return

    font, lines = _fit_text(text, width, height, draw)
    font_size = getattr(font, "size", 10)
    line_height = font_size * 1.4
    total_height = len(lines) * line_height

    start_y = y1 + (height - total_height) / 2

    for i, line in enumerate(lines):
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_width = line_bbox[2] - line_bbox[0]

        x = x1 + (width - line_width) / 2
        y = start_y + i * line_height

        draw.text((x, y), line, font=font, fill="black")


def render_translations(
    image: np.ndarray,
    regions: list[TextRegion],
    translations: list[TranslationResult],
) -> Image.Image:
    """번역 텍스트를 이미지에 렌더링"""
    if image.size == 0:
        raise RenderingError("유효하지 않은 이미지입니다")

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    trans_map = {t.index: t.translated for t in translations}

    for region in regions:
        text = trans_map.get(region.index, "")
        if not text or region.render_bbox is None:
            continue

        x1, y1, x2, y2 = region.render_bbox.to_tuple()
        _render_text_in_box(draw, text, x1, y1, x2, y2)

    logger.info(f"렌더링 완료: {len(regions)}개 영역")
    return pil_image
