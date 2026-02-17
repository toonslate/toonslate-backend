import base64
import io
import json
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient
from PIL import Image

from src.constants import RedisPrefix

if TYPE_CHECKING:
    import fakeredis

    from tests.conftest import SetupTranslateFunc


class TestErasePost:
    def test_translate_not_found(self, client: TestClient, test_mask: str) -> None:
        response = client.post(
            "/erase",
            json={"translateId": "tr_00000000", "maskImage": test_mask},
        )

        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "TRANSLATE_NOT_FOUND"

    def test_translate_not_completed(
        self,
        client: TestClient,
        test_mask: str,
        setup_translate: "SetupTranslateFunc",
    ) -> None:
        setup_translate("tr_00000001", status="pending")

        response = client.post(
            "/erase",
            json={"translateId": "tr_00000001", "maskImage": test_mask},
        )

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "TRANSLATE_NOT_COMPLETED"

    def test_success(
        self,
        client: TestClient,
        test_mask: str,
        setup_translate: "SetupTranslateFunc",
        mock_inpainting: None,
    ) -> None:
        setup_translate("tr_a1b2c3d4")

        response = client.post(
            "/erase",
            json={"translateId": "tr_a1b2c3d4", "maskImage": test_mask},
        )

        assert response.status_code == 200
        assert "resultImage" in response.json()

        result_bytes = base64.b64decode(response.json()["resultImage"])
        result_img = Image.open(io.BytesIO(result_bytes))
        assert result_img.size == (100, 100)

    def test_rgba_mask(
        self,
        client: TestClient,
        test_rgba_mask: str,
        setup_translate: "SetupTranslateFunc",
        mock_inpainting: None,
    ) -> None:
        setup_translate("tr_e5f6a7b8")

        response = client.post(
            "/erase",
            json={"translateId": "tr_e5f6a7b8", "maskImage": test_rgba_mask},
        )

        assert response.status_code == 200
        assert "resultImage" in response.json()

    # TODO: 사용자 입력 오류는 400/422가 더 적절 - INVALID_MASK_IMAGE 에러 코드 도입 검토
    def test_invalid_base64(
        self,
        client: TestClient,
        setup_translate: "SetupTranslateFunc",
    ) -> None:
        setup_translate("tr_00000002")

        response = client.post(
            "/erase",
            json={"translateId": "tr_00000002", "maskImage": "not-valid-base64!!!"},
        )

        assert response.status_code == 500
        assert response.json()["detail"]["code"] == "INPAINTING_FAILED"

    def test_invalid_redis_json(
        self,
        client: TestClient,
        test_mask: str,
        fake_redis: "fakeredis.FakeRedis",
    ) -> None:
        fake_redis.set(f"{RedisPrefix.TRANSLATE}:tr_00000003", "not valid json {{{")

        response = client.post(
            "/erase",
            json={"translateId": "tr_00000003", "maskImage": test_mask},
        )

        assert response.status_code == 500
        assert response.json()["detail"]["code"] == "INPAINTING_FAILED"

    def test_result_image_missing(
        self,
        client: TestClient,
        test_mask: str,
        fake_redis: "fakeredis.FakeRedis",
    ) -> None:
        metadata = {
            "translate_id": "tr_00000004",
            "status": "completed",
            "upload_id": "upload_test123",
            "source_language": "ko",
            "target_language": "en",
            "created_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:01:00Z",
        }
        fake_redis.set(f"{RedisPrefix.TRANSLATE}:tr_00000004", json.dumps(metadata))

        response = client.post(
            "/erase",
            json={"translateId": "tr_00000004", "maskImage": test_mask},
        )

        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "RESULT_IMAGE_NOT_FOUND"

    def test_path_traversal_blocked(self, client: TestClient, test_mask: str) -> None:
        response = client.post(
            "/erase",
            json={"translateId": "../../../etc/passwd", "maskImage": test_mask},
        )

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "INVALID_TRANSLATE_ID"

    def test_invalid_translate_id_format(self, client: TestClient, test_mask: str) -> None:
        response = client.post(
            "/erase",
            json={"translateId": "tr_ZZZZZZZZ", "maskImage": test_mask},
        )

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "INVALID_TRANSLATE_ID"

    def test_success_with_source_image(
        self,
        client: TestClient,
        test_mask: str,
        test_source_image: str,
        setup_translate: "SetupTranslateFunc",
        mock_inpainting: None,
    ) -> None:
        setup_translate("tr_a1b2c3d4")

        response = client.post(
            "/erase",
            json={
                "translateId": "tr_a1b2c3d4",
                "maskImage": test_mask,
                "sourceImage": test_source_image,
            },
        )

        assert response.status_code == 200
        assert "resultImage" in response.json()

    def test_source_image_bypasses_redis_and_file_checks(
        self,
        client: TestClient,
        test_mask: str,
        test_source_image: str,
        mock_inpainting: None,
    ) -> None:
        """sourceImage가 있으면 Redis/디스크 파일 검증을 건너뜀.

        FE가 캔버스 이미지를 보유하고 있다는 것 자체가 완료 상태의 증거.
        translate_id 형식 검증만 수행.
        """
        response = client.post(
            "/erase",
            json={
                "translateId": "tr_a1b2c3d4",
                "maskImage": test_mask,
                "sourceImage": test_source_image,
            },
        )

        assert response.status_code == 200
        assert "resultImage" in response.json()

    def test_invalid_source_image(
        self,
        client: TestClient,
        test_mask: str,
        mock_inpainting: None,
    ) -> None:
        response = client.post(
            "/erase",
            json={
                "translateId": "tr_a1b2c3d4",
                "maskImage": test_mask,
                "sourceImage": "not-valid-base64!!!",
            },
        )

        assert response.status_code == 500
        assert response.json()["detail"]["code"] == "INPAINTING_FAILED"
