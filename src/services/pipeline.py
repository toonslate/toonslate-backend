"""멀티 모델 번역 파이프라인

Detection → Inpainting → Translation → Rendering 순서로 실행.
각 단계는 Protocol 기반 모듈을 팩토리에서 가져옴.
"""

import logging

import cv2
from PIL import Image

from src.schemas.pipeline import BBox, TextRegion
from src.services.detection import get_detection
from src.services.detection.schemas import DetectionResult
from src.services.inpainting import get_inpainting
from src.services.rendering import render_translations
from src.services.translation import get_translation

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    pass


def build_text_regions(
    detection: DetectionResult,
) -> tuple[list[TextRegion], list[BBox]]:
    """DetectionResult → TextRegion 리스트 + bubble BBox 리스트 변환"""
    text_regions = [
        TextRegion(index=i, text_bbox=BBox.from_list(coords))
        for i, coords in enumerate(detection.texts)
    ]
    bubble_bboxes = [BBox.from_list(coords) for coords in detection.bubbles]
    return text_regions, bubble_bboxes


def translate_image(image_path: str) -> Image.Image:
    """이미지를 번역하여 결과 이미지 반환

    Args:
        image_path: 원본 이미지 파일 경로

    Returns:
        번역된 이미지 (PIL Image)

    Raises:
        PipelineError: 이미지 로드 실패 시
    """
    # 1. Detection
    detection = get_detection().detect(image_path)
    text_regions, bubble_bboxes = build_text_regions(detection)
    logger.info(f"Detection 완료: {len(text_regions)}개 텍스트, {len(bubble_bboxes)}개 말풍선")

    if not text_regions:
        with Image.open(image_path) as img:
            return img.convert("RGB")

    # 2. Inpainting
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise PipelineError(f"이미지를 읽을 수 없음: {image_path}")

    clean_image, updated_regions = get_inpainting().inpaint(image_bgr, text_regions, bubble_bboxes)
    logger.info(f"Inpainting 완료: {len(updated_regions)}개 영역")

    # 3. Translation
    text_bboxes = [r.text_bbox for r in text_regions]
    translations = get_translation().translate(image_path, text_bboxes)
    logger.info(f"번역 결과: {len(translations)}/{len(text_regions)}개")

    # 4. Rendering
    result = render_translations(clean_image, updated_regions, translations)
    logger.info("Rendering 완료")

    return result
