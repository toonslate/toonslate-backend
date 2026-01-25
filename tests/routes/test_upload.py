from io import BytesIO

from fastapi.testclient import TestClient


class TestUploadPost:
    def test_upload_jpeg(self, client: TestClient) -> None:
        response = client.post(
            "/upload",
            files={"file": ("test.jpg", BytesIO(b"fake jpeg"), "image/jpeg")},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["upload_id"].startswith("upload_")
        assert len(data["upload_id"]) == 15  # "upload_" + 8 chars
        assert data["filename"] == "test.jpg"
        assert data["content_type"] == "image/jpeg"
        assert "image_url" in data
        assert "created_at" in data

    def test_upload_png(self, client: TestClient) -> None:
        response = client.post(
            "/upload",
            files={"file": ("test.png", BytesIO(b"fake png"), "image/png")},
        )

        assert response.status_code == 201
        assert response.json()["content_type"] == "image/png"

    def test_reject_invalid_content_type(self, client: TestClient) -> None:
        response = client.post(
            "/upload",
            files={"file": ("test.txt", BytesIO(b"text"), "text/plain")},
        )

        assert response.status_code == 400

    def test_reject_oversized_file(self, client: TestClient) -> None:
        large_content = b"x" * (10 * 1024 * 1024 + 1)  # 10MB + 1 byte
        response = client.post(
            "/upload",
            files={"file": ("large.jpg", BytesIO(large_content), "image/jpeg")},
        )

        assert response.status_code == 400


class TestUploadGet:
    def test_get_upload(self, client: TestClient) -> None:
        # 먼저 업로드
        upload_response = client.post(
            "/upload",
            files={"file": ("test.jpg", BytesIO(b"fake jpeg"), "image/jpeg")},
        )
        upload_id = upload_response.json()["upload_id"]

        # 조회
        response = client.get(f"/upload/{upload_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["upload_id"] == upload_id
        assert data["filename"] == "test.jpg"

    def test_get_nonexistent_upload(self, client: TestClient) -> None:
        response = client.get("/upload/upload_notfound")

        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "UPLOAD_NOT_FOUND"
