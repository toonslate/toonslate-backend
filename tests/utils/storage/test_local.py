from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from src.utils.storage.local import ALLOWED_TYPES, MAX_SIZE, LocalStorage


def create_upload_file(
    content: bytes,
    filename: str = "test.jpg",
    content_type: str = "image/jpeg",
) -> UploadFile:
    return UploadFile(
        file=BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


class TestLocalStorageSave:
    async def test_save_jpeg(self, local_storage: LocalStorage) -> None:
        file = create_upload_file(b"fake jpeg content", "image.jpg", "image/jpeg")

        path = await local_storage.save(file)

        assert path.startswith("original/")
        assert path.endswith(".jpg")
        assert local_storage.exists(path)

    async def test_save_png(self, local_storage: LocalStorage) -> None:
        file = create_upload_file(b"fake png content", "image.png", "image/png")

        path = await local_storage.save(file)

        assert path.endswith(".png")
        assert local_storage.exists(path)

    async def test_save_custom_subdir(self, local_storage: LocalStorage) -> None:
        file = create_upload_file(b"content", "test.jpg", "image/jpeg")

        path = await local_storage.save(file, subdir="clean")

        assert path.startswith("clean/")

    async def test_reject_invalid_content_type(self, local_storage: LocalStorage) -> None:
        file = create_upload_file(b"not an image", "file.txt", "text/plain")

        with pytest.raises(HTTPException) as exc_info:
            await local_storage.save(file)

        assert exc_info.value.status_code == 400
        assert "지원하지 않는 파일 형식" in str(exc_info.value.detail)

    async def test_reject_none_content_type(self, local_storage: LocalStorage) -> None:
        file = UploadFile(file=BytesIO(b"content"), filename="test.jpg")

        with pytest.raises(HTTPException) as exc_info:
            await local_storage.save(file)

        assert exc_info.value.status_code == 400
        assert "알 수 없음" in str(exc_info.value.detail)

    async def test_reject_oversized_file(self, local_storage: LocalStorage) -> None:
        large_content = b"x" * (MAX_SIZE + 1)
        file = create_upload_file(large_content, "large.jpg", "image/jpeg")

        with pytest.raises(HTTPException) as exc_info:
            await local_storage.save(file)

        assert exc_info.value.status_code == 400
        assert "파일 크기 초과" in str(exc_info.value.detail)

    async def test_default_extension_when_missing(self, local_storage: LocalStorage) -> None:
        file = UploadFile(
            file=BytesIO(b"content"),
            filename="noext",
            headers=Headers({"content-type": "image/jpeg"}),
        )

        path = await local_storage.save(file)

        assert path.endswith(".jpg")


class TestLocalStorageGetUrl:
    def test_get_url(self, local_storage: LocalStorage) -> None:
        url = local_storage.get_url("original/abc123.jpg")

        assert url == "/static/original/abc123.jpg"


class TestLocalStorageExists:
    async def test_exists_true(self, local_storage: LocalStorage) -> None:
        file = create_upload_file(b"content", "test.jpg", "image/jpeg")
        path = await local_storage.save(file)

        assert local_storage.exists(path) is True

    def test_exists_false(self, local_storage: LocalStorage) -> None:
        assert local_storage.exists("nonexistent/file.jpg") is False


class TestAllowedTypes:
    def test_allowed_types_contains_jpeg_and_png(self) -> None:
        assert "image/jpeg" in ALLOWED_TYPES
        assert "image/png" in ALLOWED_TYPES

    def test_webp_not_allowed(self) -> None:
        assert "image/webp" not in ALLOWED_TYPES
