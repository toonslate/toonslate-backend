"""Detection 모듈

사용법:
    from src.services.detection import get_detection

    detector = get_detection()
    result = detector.detect(image_path)

백엔드 선택 (.env DETECTION_PROVIDER):
    - "hf_space": HuggingFace Space API (기본값)
"""

from src.config import get_settings
from src.services.detection.base import Detector
from src.services.detection.hf_space import HFSpaceDetection
from src.services.detection.schemas import DetectionResult

__all__ = ["Detector", "DetectionResult", "get_detection", "set_detection"]

# TODO: _backend → _detector 로 네이밍 변경 (translation 모듈과 통일)
_backend: Detector | None = None


def get_detection() -> Detector:
    """설정에 따라 detection 백엔드 반환"""
    global _backend
    if _backend is None:
        settings = get_settings()
        if settings.detection_provider == "hf_space":
            _backend = HFSpaceDetection(
                space_url=settings.hf_space_url,
                api_timeout=settings.hf_api_timeout,
            )
        else:
            raise ValueError(f"Unknown detection provider: {settings.detection_provider!r}")
    return _backend


def set_detection(backend: Detector | None) -> None:
    """detection 백엔드 설정 (테스트용)"""
    global _backend
    _backend = backend
