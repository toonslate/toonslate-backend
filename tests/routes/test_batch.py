from unittest.mock import ANY, MagicMock, patch

from fastapi.testclient import TestClient

from src.constants import Limits
from tests.conftest import make_test_image


def _upload_image(client: TestClient) -> str:
    response = client.post(
        "/upload",
        files={"file": ("test.jpg", make_test_image(), "image/jpeg")},
    )
    return response.json()["uploadId"]


class TestCreateBatch:
    def test_success(self, client: TestClient) -> None:
        ids = [_upload_image(client) for _ in range(3)]

        with patch("src.routes.batch.translate_job"):
            response = client.post("/batch", json={"uploadIds": ids})

        assert response.status_code == 201
        data = response.json()
        assert data["batchId"].startswith("batch_")
        assert data["status"] == "processing"
        assert len(data["images"]) == 3

    def test_invalid_upload_id(self, client: TestClient) -> None:
        with patch("src.routes.batch.translate_job"):
            response = client.post(
                "/batch",
                json={"uploadIds": ["nonexistent_upload_id"]},
            )

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "INVALID_UPLOAD_ID"

    def test_empty_upload_ids(self, client: TestClient) -> None:
        response = client.post("/batch", json={"uploadIds": []})

        assert response.status_code == 422

    def test_exceeds_max_batch_size(self, client: TestClient) -> None:
        ids = [f"upload_{i}" for i in range(Limits.MAX_BATCH_SIZE + 1)]
        response = client.post("/batch", json={"uploadIds": ids})

        assert response.status_code == 422

    def test_all_queuing_failed_returns_503(self, client: TestClient) -> None:
        ids = [_upload_image(client) for _ in range(2)]
        mock_job = MagicMock()
        mock_job.delay.side_effect = RuntimeError("broker down")

        with (
            patch("src.routes.batch.translate_job", mock_job),
            patch("src.routes.batch.refund_quota") as mock_refund,
        ):
            response = client.post("/batch", json={"uploadIds": ids})

        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "QUEUE_UNAVAILABLE"
        mock_refund.assert_awaited_once_with(ANY, 2)

    def test_rate_limit_exceeded(self, client: TestClient) -> None:
        batch_size = Limits.MAX_BATCH_SIZE
        batch_1 = [_upload_image(client) for _ in range(batch_size)]
        batch_2 = [_upload_image(client) for _ in range(batch_size)]

        with patch("src.routes.batch.translate_job"):
            resp1 = client.post("/batch", json={"uploadIds": batch_1})
            assert resp1.status_code == 201
            resp2 = client.post("/batch", json={"uploadIds": batch_2})
            assert resp2.status_code == 201

        extra = [_upload_image(client)]
        with patch("src.routes.batch.translate_job"):
            response = client.post("/batch", json={"uploadIds": extra})

        assert response.status_code == 429
        assert response.json()["detail"]["code"] == "RATE_LIMIT_EXCEEDED"


class TestGetBatch:
    def test_success(self, client: TestClient) -> None:
        ids = [_upload_image(client)]

        with patch("src.routes.batch.translate_job"):
            create_resp = client.post("/batch", json={"uploadIds": ids})

        batch_id = create_resp.json()["batchId"]
        response = client.get(f"/batch/{batch_id}")

        assert response.status_code == 200
        assert response.json()["batchId"] == batch_id

    def test_not_found(self, client: TestClient) -> None:
        response = client.get("/batch/batch_nonexist")

        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "BATCH_NOT_FOUND"
