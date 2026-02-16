"""Inpainting 모듈

사용법:
    from src.services.inpainting import get_inpainting

    inpainter = get_inpainting()
    clean_image, regions = inpainter.inpaint(image, text_regions, bubble_bboxes)

백엔드 선택 (.env INPAINTING_PROVIDER):
    - "iopaint_lama": RoutedInpainting (말풍선→SolidFill, 배경→IOPaint LaMa)
    - "replicate_lama": Replicate LaMa API (레거시)
    - "solid_fill": 단색 채우기 (레거시, 기본값)
"""

from src.config import get_settings
from src.services.inpainting.base import Inpainter
from src.services.inpainting.solid_fill import InpaintingError

__all__ = ["Inpainter", "InpaintingError", "get_inpainting", "set_inpainting"]

_inpainter: Inpainter | None = None


def get_inpainting() -> Inpainter:
    """설정에 따라 inpainting 백엔드 반환"""
    global _inpainter
    if _inpainter is None:
        settings = get_settings()
        if settings.inpainting_provider == "iopaint_lama":
            _inpainter = _create_routed_inpainting()
        elif settings.inpainting_provider == "replicate_lama":
            from src.services.inpainting.replicate_lama import ReplicateLamaInpainting

            _inpainter = ReplicateLamaInpainting(
                settings.replicate_api_token,
                debug_dir=settings.inpainting_debug_dir or None,
            )
        else:
            from src.services.inpainting.solid_fill import SolidFillInpainting

            _inpainter = SolidFillInpainting()
    return _inpainter


def set_inpainting(inpainter: Inpainter | None) -> None:
    """inpainting 백엔드 설정 (테스트용)"""
    global _inpainter
    _inpainter = inpainter


def _create_routed_inpainting() -> Inpainter:
    from src.services.inpainting.background_restorer import IOPaintRestorer
    from src.services.inpainting.bubble_cleaner import SolidFillBubbleCleaner
    from src.services.inpainting.classifier import RegionClassifier
    from src.services.inpainting.inpainter import RoutedInpainting

    settings = get_settings()
    return RoutedInpainting(
        classifier=RegionClassifier(),
        bubble_cleaner=SolidFillBubbleCleaner(),
        background_restorer=IOPaintRestorer(
            space_url=settings.iopaint_space_url,
            timeout=settings.iopaint_timeout,
        ),
    )
