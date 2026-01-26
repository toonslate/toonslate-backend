import tempfile
from collections.abc import Generator
from io import BytesIO
from pathlib import Path

import fakeredis
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.utils.redis import set_redis
from src.utils.storage import set_storage
from src.utils.storage.local import LocalStorage


@pytest.fixture
def temp_upload_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def local_storage(temp_upload_dir: Path) -> LocalStorage:
    return LocalStorage(base_dir=temp_upload_dir, base_url="/static")


@pytest.fixture
def fake_redis() -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis(decode_responses=True)


@pytest.fixture
def client(
    temp_upload_dir: Path, fake_redis: fakeredis.FakeAsyncRedis
) -> Generator[TestClient, None, None]:
    storage = LocalStorage(base_dir=temp_upload_dir, base_url="http://localhost:8000/static")
    set_storage(storage)
    set_redis(fake_redis)
    yield TestClient(app)


@pytest.fixture
def upload_id(client: TestClient) -> str:
    response = client.post(
        "/upload",
        files={"file": ("test.jpg", BytesIO(b"fake jpeg"), "image/jpeg")},
    )
    return response.json()["uploadId"]


@pytest.fixture
def job_id(client: TestClient, upload_id: str) -> str:
    response = client.post(
        "/jobs",
        json={
            "uploadIds": [upload_id],
            "options": {"sourceLanguage": "ko", "targetLanguage": "en"},
        },
    )
    return response.json()["jobId"]
