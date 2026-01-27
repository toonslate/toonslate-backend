import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_SIZE = 10 * 1024 * 1024  # 10MB
CHUNK_SIZE = 1024 * 1024  # 1MB


class LocalStorage:
    """로컬 파일 시스템 저장소 구현체. S3Storage로 교체 가능."""

    def __init__(self, base_dir: Path, base_url: str = "/static"):
        self.base_dir = base_dir
        self.base_url = base_url

    async def save(
        self, file: UploadFile, subdir: str = "original", filename: str | None = None
    ) -> str:
        """
        Raises:
            HTTPException(400): 지원하지 않는 파일 형식 또는 10MB 초과 시
        """
        self._validate_content_type(file.content_type)
        self._validate_size_header(file.size)

        content = await self._read_with_size_limit(file)

        name = filename or uuid.uuid4().hex
        ext = Path(file.filename or "").suffix or ".jpg"

        relative_path = f"{subdir}/{name}{ext}"
        save_path = self.base_dir / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_path.write_bytes(content)
        return relative_path

    def get_url(self, relative_path: str) -> str:
        return f"{self.base_url}/{relative_path}"

    def get_absolute_path(self, relative_path: str) -> str:
        return str(self.base_dir / relative_path)

    def exists(self, relative_path: str) -> bool:
        return (self.base_dir / relative_path).exists()

    def delete(self, relative_path: str) -> bool:
        file_path = self.base_dir / relative_path
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def _validate_content_type(self, content_type: str | None) -> None:
        if not content_type or content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 파일 형식: {content_type or '알 수 없음'}",
            )

    def _validate_size_header(self, size: int | None) -> None:
        if size is not None and size > MAX_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"파일 크기 초과: {size} bytes (최대 {MAX_SIZE} bytes)",
            )

    async def _read_with_size_limit(self, file: UploadFile) -> bytes:
        chunks: list[bytes] = []
        total_size = 0

        while chunk := await file.read(CHUNK_SIZE):
            total_size += len(chunk)
            if total_size > MAX_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"파일 크기 초과: {total_size}+ bytes (최대 {MAX_SIZE} bytes)",
                )
            chunks.append(chunk)

        return b"".join(chunks)
