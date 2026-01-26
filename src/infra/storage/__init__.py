from pathlib import Path

from .base import StorageBackend
from .local import LocalStorage

__all__ = ["StorageBackend", "LocalStorage", "get_storage", "set_storage"]


def _find_project_root() -> Path:
    """pyproject.toml 위치를 프로젝트 루트로 탐색"""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("프로젝트 루트를 찾을 수 없음")


class _StorageHolder:
    instance: StorageBackend | None = None


def get_storage() -> StorageBackend:
    if _StorageHolder.instance is None:
        upload_dir = _find_project_root() / "uploads"
        _StorageHolder.instance = LocalStorage(base_dir=upload_dir)
    return _StorageHolder.instance


def set_storage(storage: StorageBackend) -> None:
    _StorageHolder.instance = storage
