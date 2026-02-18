import base64
import json
import tempfile
from collections.abc import Generator
from io import BytesIO
from pathlib import Path
from typing import Protocol

import fakeredis
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.constants import RedisPrefix
from src.infra.redis import set_redis
from src.infra.storage import set_storage
from src.infra.storage.local import LocalStorage
from src.main import app


class SetupTranslateFunc(Protocol):
    def __call__(self, translate_id: str, status: str = "completed") -> None: ...


def make_test_image(width: int = 800, height: int = 1200, fmt: str = "JPEG") -> BytesIO:
    """테스트용 실제 이미지 바이트 생성"""
    img = Image.new("RGB", (width, height), color="red")
    buf = BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


@pytest.fixture
def temp_upload_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def local_storage(temp_upload_dir: Path) -> LocalStorage:
    return LocalStorage(base_dir=temp_upload_dir, base_url="/static")


@pytest.fixture
def fake_redis() -> Generator[fakeredis.FakeRedis, None, None]:
    r = fakeredis.FakeRedis(decode_responses=True)
    set_redis(r)
    yield r
    set_redis(None)


@pytest.fixture
def client(
    temp_upload_dir: Path, fake_redis: fakeredis.FakeRedis
) -> Generator[TestClient, None, None]:
    storage = LocalStorage(base_dir=temp_upload_dir, base_url="http://localhost:8000/static")
    set_storage(storage)
    yield TestClient(app)
    set_storage(None)


@pytest.fixture
def upload_id(client: TestClient) -> str:
    response = client.post(
        "/upload",
        files={"file": ("test.jpg", make_test_image(), "image/jpeg")},
    )
    return response.json()["uploadId"]


@pytest.fixture
def test_mask() -> str:
    """100x100 흰색 마스크 (base64)"""
    img = Image.new("L", (100, 100), color=255)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def test_rgba_mask() -> str:
    """100x100 RGBA 마스크 (base64)"""
    img = Image.new("RGBA", (100, 100), color=(255, 255, 255, 255))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def test_source_image() -> str:
    """100x100 RGB 소스 이미지 (base64)"""
    img = Image.new("RGB", (100, 100), color="blue")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def setup_translate(
    fake_redis: fakeredis.FakeRedis,
    temp_upload_dir: Path,
) -> Generator[SetupTranslateFunc, None, None]:
    """번역 메타데이터 + 결과 이미지 설정 팩토리

    사용법:
        setup_translate("tr_a1b2c3d4")  # completed 상태
        setup_translate("tr_a1b2c3d4", status="pending")  # pending 상태
    """

    def _setup(translate_id: str, status: str = "completed") -> None:
        metadata = {
            "translate_id": translate_id,
            "status": status,
            "upload_id": "upload_test123",
            "source_language": "ko",
            "target_language": "en",
            "created_at": "2026-01-01T00:00:00Z",
        }
        if status == "completed":
            metadata["completed_at"] = "2026-01-01T00:01:00Z"

        fake_redis.set(f"{RedisPrefix.TRANSLATE}:{translate_id}", json.dumps(metadata))

        if status == "completed":
            result_dir = temp_upload_dir / "result"
            result_dir.mkdir(parents=True, exist_ok=True)
            result_path = result_dir / f"{translate_id}_result.png"

            img = Image.new("RGB", (100, 100), color="red")
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            result_path.write_bytes(buffer.getvalue())

    yield _setup


@pytest.fixture
def mock_inpainting() -> Generator[None, None, None]:
    """Inpainting 백엔드 mock"""
    from src.services.inpainting import set_inpainting

    class MockInpainting:
        def inpaint_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
            result = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
            result[:, :] = [0, 255, 0]
            return result

        def inpaint(
            self,
            image: np.ndarray,
            text_regions: list[object],
            bubble_bboxes: list[object],
        ) -> tuple[np.ndarray, list[object]]:
            raise NotImplementedError()

    set_inpainting(MockInpainting())  # type: ignore[arg-type]
    yield
    set_inpainting(None)  # type: ignore[arg-type]
