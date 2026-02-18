import uuid
from io import BytesIO
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image

ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_SIZE = 5 * 1024 * 1024  # 5MB
CHUNK_SIZE = 1024 * 1024  # 1MB

MAGIC_BYTES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG": "image/png",
}
MIN_WIDTH = 600
MAX_PIXELS = 3_000_000
MAX_ASPECT_RATIO = 3.0


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
            HTTPException(400): 파일 형식, 크기, 이미지 규격 위반 시
        """
        self._validate_content_type(file.content_type)
        self._validate_size_header(file.size)

        content = await self._read_with_size_limit(file)
        detected_type = self._detect_image_type(content)
        self._validate_content_type_match(detected_type, file.content_type)
        self._validate_image_dimensions(content)

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

    def _detect_image_type(self, content: bytes) -> str:
        for magic, mime in MAGIC_BYTES.items():
            if content.startswith(magic):
                return mime
        raise HTTPException(status_code=400, detail="유효하지 않은 이미지 파일")

    def _validate_content_type_match(self, detected: str, declared: str | None) -> None:
        if declared and detected != declared:
            raise HTTPException(
                status_code=400,
                detail=f"파일 형식 불일치: 헤더 {declared}, 실제 {detected}",
            )

    def _validate_image_dimensions(self, content: bytes) -> None:
        try:
            img = Image.open(BytesIO(content))
            width, height = img.size
        except Exception as e:
            raise HTTPException(status_code=400, detail="이미지 디코딩 실패") from e

        if width < MIN_WIDTH:
            raise HTTPException(
                status_code=400,
                detail=f"이미지 너비 부족: {width}px (최소 {MIN_WIDTH}px)",
            )

        if width * height > MAX_PIXELS:
            raise HTTPException(
                status_code=400,
                detail=f"총 픽셀수 초과: {width}x{height} = {width * height} (최대 {MAX_PIXELS})",
            )

        if height / width > MAX_ASPECT_RATIO:
            raise HTTPException(
                status_code=400,
                detail=f"세로/가로 비율 초과: {height / width:.2f} (최대 {MAX_ASPECT_RATIO})",
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
