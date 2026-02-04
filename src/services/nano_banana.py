"""Nano Banana: Gemini 3 Pro Image 기반 웹툰 번역 서비스

Gemini Image Generation API를 사용하여 웹툰 이미지의 한국어 텍스트를
영어로 번역된 이미지로 변환합니다.

제약사항:
- 출력 해상도 최대 768x1344px (9:16)
- 다중 출력 미지원 (SDK 제한)
- 따라서 긴 이미지는 ~1200px 단위로 분할 후 개별 API 호출

TODO: 알려진 문제 (docs/tasks/nano-banana-improvements.md 참고)
- 검은 배경 이미지에서 분할 실패 (밝은 여백만 감지)
- 세그먼트 수 제한 필요 (타임아웃, 비용 관리)
- Rate limit 대응 필요 (RPM=20, RPD=250)
"""

import io
import logging

from google import genai
from google.genai import types
from PIL import Image, ImageStat

from src.config import get_settings

logger = logging.getLogger(__name__)

GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"

# 출력 1K 기준 (9:16 aspect ratio → 최대 1344px 높이)
# 축소 배율 2배 이상이면 텍스트 품질 저하 → 분할 필요
EXPECTED_OUTPUT_HEIGHT = 1344
MAX_SCALE_RATIO = 2.0
MAX_SEGMENT_HEIGHT = int(EXPECTED_OUTPUT_HEIGHT * MAX_SCALE_RATIO)  # 2688
MIN_SEGMENT_HEIGHT = int(MAX_SEGMENT_HEIGHT * 0.6)  # 1612
WHITESPACE_THRESHOLD = 240

TRANSLATE_PROMPT = (
    "Change all Korean text in this webtoon to English "
    "so that English-speaking readers can fully understand the content. "
    "Keep everything else exactly the same."
)


class NanoBananaError(Exception):
    pass


def create_client() -> genai.Client:
    """Gemini API 클라이언트 생성"""
    settings = get_settings()
    if not settings.gemini_api_key:
        raise NanoBananaError("GEMINI_API_KEY가 설정되지 않았습니다")
    return genai.Client(api_key=settings.gemini_api_key)


def find_split_point(image: Image.Image, start_y: int, end_y: int) -> int:
    """주어진 범위 내에서 가장 밝은 가로 행(여백)을 찾아 분할 지점 반환

    Args:
        image: 흑백 변환된 이미지
        start_y: 탐색 시작 y좌표
        end_y: 탐색 끝 y좌표

    Returns:
        분할 지점 y좌표 (여백을 찾지 못하면 end_y 반환)
    """
    width = image.width
    best_y = end_y
    best_brightness: float = 0

    for y in range(end_y, start_y, -1):
        row = image.crop((0, y - 2, width, y + 3))
        stat = ImageStat.Stat(row)
        brightness: float = stat.mean[0]

        if brightness >= WHITESPACE_THRESHOLD:
            return y

        if brightness > best_brightness:
            best_brightness = brightness
            best_y = y

    return best_y


def split_image(image: Image.Image) -> list[Image.Image]:
    """긴 웹툰 이미지를 여러 세그먼트로 분할

    여백(밝은 가로 행)을 기준으로 자연스러운 분할점을 찾습니다.
    출력 해상도 제한(1344px)에 맞추기 위해 ~1200px 단위로 분할.

    Args:
        image: 원본 이미지 (RGB)

    Returns:
        분할된 이미지 리스트
    """
    height = image.height

    if height <= MAX_SEGMENT_HEIGHT:
        return [image]

    grayscale = image.convert("L")
    segments: list[Image.Image] = []
    current_y = 0

    while current_y < height:
        remaining = height - current_y

        if remaining <= MAX_SEGMENT_HEIGHT:
            segments.append(image.crop((0, current_y, image.width, height)))
            break

        search_start = current_y + MIN_SEGMENT_HEIGHT
        search_end = min(current_y + MAX_SEGMENT_HEIGHT, height)
        split_y = find_split_point(grayscale, search_start, search_end)

        segments.append(image.crop((0, current_y, image.width, split_y)))
        current_y = split_y

    logger.info(f"이미지 분할: {height}px → {len(segments)}개 세그먼트")
    return segments


def merge_images(segments: list[Image.Image]) -> Image.Image:
    """분할된 이미지들을 세로로 합침

    Args:
        segments: 분할된 이미지 리스트

    Returns:
        합쳐진 이미지
    """
    if len(segments) == 1:
        return segments[0]

    width = segments[0].width
    total_height = sum(seg.height for seg in segments)
    merged = Image.new("RGB", (width, total_height))

    current_y = 0
    for segment in segments:
        merged.paste(segment, (0, current_y))
        current_y += segment.height

    return merged


def translate_segment(client: genai.Client, image: Image.Image) -> Image.Image:
    """단일 세그먼트 번역

    Args:
        client: Gemini API 클라이언트
        image: 번역할 이미지 세그먼트

    Returns:
        번역된 이미지

    Raises:
        NanoBananaError: API 호출 실패 또는 응답에 이미지 없음
    """
    try:
        response = client.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=[TRANSLATE_PROMPT, image],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )
    except Exception as e:
        raise NanoBananaError(f"Gemini API 호출 실패: {e}") from e

    if response.parts is None:
        raise NanoBananaError("응답에 parts가 없습니다")

    for part in response.parts:
        if part.inline_data is not None and part.inline_data.data is not None:
            image_bytes = part.inline_data.data
            result = Image.open(io.BytesIO(image_bytes))
            return result.convert("RGB")

    raise NanoBananaError("응답에 이미지가 없습니다")


def translate_image(image_path: str) -> Image.Image:
    """웹툰 이미지 번역 (자동 분할/병합)

    긴 이미지는 자동으로 분할하여 각각 번역 후 다시 합칩니다.

    Args:
        image_path: 원본 이미지 경로

    Returns:
        번역된 이미지 (PIL Image)

    Raises:
        NanoBananaError: API 키 미설정, API 호출 실패 등
    """
    client = create_client()

    with Image.open(image_path) as original:
        image = original.convert("RGB")
        segments = split_image(image)

    translated: list[Image.Image] = []

    for i, segment in enumerate(segments):
        logger.info(f"세그먼트 번역 중: {i + 1}/{len(segments)}")
        result = translate_segment(client, segment)
        translated.append(result)

    merged = merge_images(translated)
    logger.info(f"번역 완료: {len(segments)}개 세그먼트 처리")

    return merged
