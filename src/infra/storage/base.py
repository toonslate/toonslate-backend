from typing import Protocol

from fastapi import UploadFile


class StorageBackend(Protocol):
    """파일 저장소 인터페이스. LocalStorage, S3Storage 등 구현체로 교체 가능."""

    async def save(
        self, file: UploadFile, subdir: str = "original", filename: str | None = None
    ) -> str: ...
    def get_url(self, relative_path: str) -> str: ...
    def exists(self, relative_path: str) -> bool: ...
    def delete(self, relative_path: str) -> bool: ...
