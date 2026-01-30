"""Gemini 기반 번역 서비스"""

import io
import json
import logging
from typing import cast

from google import genai
from google.genai import types
from PIL import Image

from src.config import get_settings
from src.schemas.pipeline import BBox, TranslationResult

logger = logging.getLogger(__name__)

TRANSLATE_PROMPT = """각 이미지는 웹툰/만화에서 크롭한 텍스트 영역입니다.
각 이미지의 한국어 텍스트를 영어로 번역해주세요.

규칙:
- 이미지 순서대로 index 부여 (0부터 시작)
- 의성어/의태어는 자연스러운 영어 효과음으로 번역
- 텍스트가 없거나 읽을 수 없으면 translated를 빈 문자열로

JSON 배열로만 응답:
[{"index": 0, "translated": "Hello"}, {"index": 1, "translated": "BOOM"}, ...]"""


class TranslationError(Exception):
    pass


def _create_client() -> genai.Client:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise TranslationError("GEMINI_API_KEY가 설정되지 않았습니다")
    return genai.Client(api_key=settings.gemini_api_key)


def _crop_to_parts(image_path: str, bboxes: list[BBox]) -> tuple[list[types.Part], list[int]]:
    """이미지를 크롭하여 Gemini Part 리스트로 변환. 유효한 bbox의 원본 인덱스도 반환."""
    parts: list[types.Part] = []
    original_indices: list[int] = []

    with Image.open(image_path) as image:
        for idx, bbox in enumerate(bboxes):
            if not bbox.is_valid():
                logger.warning(f"유효하지 않은 bbox 스킵: index={idx}")
                continue

            cropped = image.crop(bbox.to_tuple())
            buffer = io.BytesIO()
            cropped.save(buffer, format="PNG")
            parts.append(types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png"))
            original_indices.append(idx)

    return parts, original_indices


def _call_gemini(client: genai.Client, parts: list[types.Part]) -> list[dict[str, object]]:
    settings = get_settings()
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[TRANSLATE_PROMPT, *parts],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    if not response.text:
        raise TranslationError("빈 응답")

    try:
        raw_results = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise TranslationError(f"JSON 파싱 실패: {e}") from e

    if not isinstance(raw_results, list):
        raise TranslationError(f"응답이 리스트가 아님: {type(raw_results).__name__}")

    return cast(list[dict[str, object]], raw_results)


def _map_results(
    raw_results: list[dict[str, object]], original_indices: list[int]
) -> list[TranslationResult]:
    """Gemini 응답의 parts 인덱스를 원본 bbox 인덱스로 변환"""
    results: list[TranslationResult] = []

    for item in raw_results:
        try:
            parsed = TranslationResult.model_validate(item)
            parts_idx = parsed.index

            if 0 <= parts_idx < len(original_indices):
                original_idx = original_indices[parts_idx]
                results.append(TranslationResult(index=original_idx, translated=parsed.translated))
        except Exception as e:
            logger.warning(f"번역 결과 파싱 실패: {item} - {e}")

    results.sort(key=lambda r: r.index)
    return results


def translate_regions(
    image_path: str,
    bboxes: list[BBox],
) -> list[TranslationResult]:
    """텍스트 영역들을 한 번의 API 호출로 번역"""
    if not bboxes:
        return []

    client = _create_client()
    parts, original_indices = _crop_to_parts(image_path, bboxes)

    if not parts:
        return []

    raw_results = _call_gemini(client, parts)
    results = _map_results(raw_results, original_indices)

    logger.info(f"번역 완료: {len(results)}/{len(bboxes)}개")
    return results
