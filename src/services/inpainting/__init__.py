"""Inpainting 모듈

사용법:
    from src.services.inpainting import get_inpainting

    backend = get_inpainting()
    clean_image, regions = backend.inpaint(image, text_regions, bubble_bboxes)

백엔드 선택 (.env INPAINTING_PROVIDER):
    - "solid_fill": 단색 채우기 (기본값)
    - "replicate_lama": Replicate LaMa API (유료)
    - "iopaint_lama": IOPaint HuggingFace Space (무료)
"""

from src.config import get_settings
from src.services.inpainting.base import InpaintingBackend
from src.services.inpainting.solid_fill import InpaintingError, SolidFillInpainting

__all__ = ["InpaintingBackend", "InpaintingError", "get_inpainting", "set_inpainting"]

_backend: InpaintingBackend | None = None


def get_inpainting() -> InpaintingBackend:
    """설정에 따라 inpainting 백엔드 반환"""
    global _backend
    if _backend is None:
        settings = get_settings()
        if settings.inpainting_provider == "replicate_lama":
            from src.services.inpainting.replicate_lama import ReplicateLamaInpainting

            _backend = ReplicateLamaInpainting(
                settings.replicate_api_token,
                debug_dir=settings.inpainting_debug_dir or None,
            )
        elif settings.inpainting_provider == "iopaint_lama":
            from src.services.inpainting.iopaint_lama import IOPaintLamaInpainting

            _backend = IOPaintLamaInpainting(
                space_url=settings.iopaint_space_url,
                timeout=settings.iopaint_timeout,
                debug_dir=settings.inpainting_debug_dir or None,
            )
        else:
            _backend = SolidFillInpainting()
    return _backend


def set_inpainting(backend: InpaintingBackend) -> None:
    """inpainting 백엔드 설정

    Args:
        backend: InpaintingBackend 프로토콜을 구현한 인스턴스
    """
    global _backend
    _backend = backend
