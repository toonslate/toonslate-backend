from fastapi.testclient import TestClient


class TestJobPost:
    def test_create_job(self, client: TestClient, upload_id: str) -> None:
        response = client.post(
            "/jobs",
            json={
                "uploadIds": [upload_id],
                "options": {"sourceLanguage": "ko", "targetLanguage": "en"},
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["jobId"].startswith("job_")
        assert len(data["jobId"]) == 12
        assert data["status"] == "pending"
        assert data["uploadIds"] == [upload_id]
        assert data["options"]["sourceLanguage"] == "ko"
        assert data["options"]["targetLanguage"] == "en"
        assert "createdAt" in data

    def test_create_job_invalid_upload_id(self, client: TestClient) -> None:
        response = client.post(
            "/jobs",
            json={
                "uploadIds": ["upload_notfound"],
                "options": {"sourceLanguage": "ko", "targetLanguage": "en"},
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "INVALID_UPLOAD_ID"
        assert response.json()["detail"]["uploadId"] == "upload_notfound"

    def test_create_job_rate_limit(self, client: TestClient, upload_id: str) -> None:
        for _ in range(10):
            response = client.post(
                "/jobs",
                json={
                    "uploadIds": [upload_id],
                    "options": {"sourceLanguage": "ko", "targetLanguage": "en"},
                },
            )
            assert response.status_code == 201

        response = client.post(
            "/jobs",
            json={
                "uploadIds": [upload_id],
                "options": {"sourceLanguage": "ko", "targetLanguage": "en"},
            },
        )

        assert response.status_code == 429
        assert response.json()["detail"]["code"] == "RATE_LIMIT_EXCEEDED"


class TestJobGet:
    def test_read_job(self, client: TestClient, job_id: str) -> None:
        response = client.get(f"/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["jobId"] == job_id
        assert data["status"] == "pending"

    def test_read_nonexistent_job(self, client: TestClient) -> None:
        response = client.get("/jobs/job_notfound")

        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "JOB_NOT_FOUND"
