"""Inpainting 모듈

사용법:
    from src.services.inpainting import get_inpainting

    backend = get_inpainting()
    clean_image, regions = backend.inpaint(image, text_regions, bubble_bboxes)

백엔드 교체:
    from src.services.inpainting import set_inpainting
    from src.services.inpainting.replicate_lama import ReplicateLamaInpainting

    set_inpainting(ReplicateLamaInpainting())
"""

from src.services.inpainting.base import InpaintingBackend
from src.services.inpainting.solid_fill import InpaintingError, SolidFillInpainting

__all__ = ["InpaintingBackend", "InpaintingError", "get_inpainting", "set_inpainting"]

_backend: InpaintingBackend | None = None


def get_inpainting() -> InpaintingBackend:
    """현재 설정된 inpainting 백엔드 반환

    기본값: SolidFillInpainting
    """
    global _backend
    if _backend is None:
        _backend = SolidFillInpainting()
    return _backend


def set_inpainting(backend: InpaintingBackend) -> None:
    """inpainting 백엔드 설정

    Args:
        backend: InpaintingBackend 프로토콜을 구현한 인스턴스
    """
    global _backend
    _backend = backend
