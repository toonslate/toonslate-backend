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


_storage: StorageBackend | None = None


def get_storage() -> StorageBackend:
    global _storage
    if _storage is None:
        upload_dir = _find_project_root() / "uploads"
        _storage = LocalStorage(base_dir=upload_dir)
    return _storage


def set_storage(storage: StorageBackend) -> None:
    global _storage
    _storage = storage
