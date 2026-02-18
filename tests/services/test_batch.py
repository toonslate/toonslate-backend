import fakeredis
import pytest
from pydantic import ValidationError

from src.constants import TTL, Limits
from src.services.batch import BatchRequest, create_batch, get_batch
from src.services.translate import update_translate_status

URLS_1 = ["http://example.com/aaa.jpg"]
URLS_3 = [
    "http://example.com/aaa.jpg",
    "http://example.com/bbb.jpg",
    "http://example.com/ccc.jpg",
]


class TestBatchRequestValidation:
    def test_empty_upload_ids_raises(self) -> None:
        with pytest.raises(ValidationError, match="upload_ids"):
            BatchRequest(upload_ids=[])

    def test_exceeds_max_batch_size_raises(self) -> None:
        ids = [f"upload_{i}" for i in range(Limits.MAX_BATCH_SIZE + 1)]
        with pytest.raises(ValidationError, match="upload_ids"):
            BatchRequest(upload_ids=ids)

    def test_max_batch_size_accepted(self) -> None:
        ids = [f"upload_{i}" for i in range(Limits.MAX_BATCH_SIZE)]
        request = BatchRequest(upload_ids=ids)
        assert len(request.upload_ids) == Limits.MAX_BATCH_SIZE


class TestCreateBatch:
    async def test_success(self, fake_redis: fakeredis.FakeRedis) -> None:
        request = BatchRequest(
            upload_ids=["upload_aaa", "upload_bbb", "upload_ccc"],
            source_language="ko",
            target_language="en",
        )

        response = await create_batch(request, URLS_3)

        assert response.batch_id.startswith("batch_")
        assert response.status == "processing"
        assert len(response.images) == 3
        assert response.source_language == "ko"
        assert response.target_language == "en"

    async def test_preserves_order(self, fake_redis: fakeredis.FakeRedis) -> None:
        request = BatchRequest(
            upload_ids=["upload_aaa", "upload_bbb", "upload_ccc"],
        )

        response = await create_batch(request, URLS_3)

        for i, image in enumerate(response.images):
            assert image.order_index == i
            assert image.upload_id == request.upload_ids[i]

    async def test_creates_individual_translates(self, fake_redis: fakeredis.FakeRedis) -> None:
        request = BatchRequest(upload_ids=["upload_aaa", "upload_bbb"])

        response = await create_batch(request, URLS_3[:2])

        for image in response.images:
            assert image.translate_id.startswith("tr_")
            assert fake_redis.exists(f"translate:{image.translate_id}")

    async def test_individual_translates_start_pending(
        self, fake_redis: fakeredis.FakeRedis
    ) -> None:
        request = BatchRequest(upload_ids=["upload_aaa"])

        response = await create_batch(request, URLS_1)

        assert response.images[0].status == "pending"

    async def test_batch_metadata_has_ttl(self, fake_redis: fakeredis.FakeRedis) -> None:
        request = BatchRequest(upload_ids=["upload_aaa"])

        response = await create_batch(request, URLS_1)

        # fakeredis 타입 스텁 제한으로 int() 래핑 필요
        ttl = int(fake_redis.ttl(f"batch:{response.batch_id}"))  # type: ignore[reportArgumentType]
        assert 7100 < ttl <= TTL.DATA

    async def test_urls_length_mismatch_raises(self, fake_redis: fakeredis.FakeRedis) -> None:
        request = BatchRequest(upload_ids=["upload_aaa", "upload_bbb"])

        with pytest.raises(ValueError):
            await create_batch(request, URLS_1)


class TestGetBatch:
    async def test_not_found(self, fake_redis: fakeredis.FakeRedis) -> None:
        result = await get_batch("batch_nonexist")
        assert result is None

    async def test_initial_status_is_processing(self, fake_redis: fakeredis.FakeRedis) -> None:
        request = BatchRequest(upload_ids=["upload_aaa", "upload_bbb"])
        created = await create_batch(request, URLS_3[:2])

        result = await get_batch(created.batch_id)

        assert result is not None
        assert result.status == "processing"

    async def test_all_completed(self, fake_redis: fakeredis.FakeRedis) -> None:
        request = BatchRequest(upload_ids=["upload_aaa", "upload_bbb"])
        created = await create_batch(request, URLS_3[:2])

        for image in created.images:
            await update_translate_status(image.translate_id, "completed")

        result = await get_batch(created.batch_id)

        assert result is not None
        assert result.status == "completed"

    async def test_all_failed(self, fake_redis: fakeredis.FakeRedis) -> None:
        request = BatchRequest(upload_ids=["upload_aaa", "upload_bbb"])
        created = await create_batch(request, URLS_3[:2])

        for image in created.images:
            await update_translate_status(image.translate_id, "failed", "error")

        result = await get_batch(created.batch_id)

        assert result is not None
        assert result.status == "failed"

    async def test_partial_failure(self, fake_redis: fakeredis.FakeRedis) -> None:
        request = BatchRequest(upload_ids=["upload_aaa", "upload_bbb"])
        created = await create_batch(request, URLS_3[:2])

        await update_translate_status(created.images[0].translate_id, "completed")
        await update_translate_status(created.images[1].translate_id, "failed", "error msg")

        result = await get_batch(created.batch_id)

        assert result is not None
        assert result.status == "partial_failure"

    async def test_processing_overrides_partial_failure(
        self, fake_redis: fakeredis.FakeRedis
    ) -> None:
        request = BatchRequest(upload_ids=["upload_aaa", "upload_bbb", "upload_ccc"])
        created = await create_batch(request, URLS_3)

        await update_translate_status(created.images[0].translate_id, "completed")
        await update_translate_status(created.images[1].translate_id, "failed", "error")

        result = await get_batch(created.batch_id)

        assert result is not None
        assert result.status == "processing"

    async def test_reflects_completed_translate_fields(
        self, fake_redis: fakeredis.FakeRedis
    ) -> None:
        request = BatchRequest(upload_ids=["upload_aaa"])
        created = await create_batch(request, URLS_1)

        await update_translate_status(created.images[0].translate_id, "completed")

        result = await get_batch(created.batch_id)

        assert result is not None
        assert result.images[0].status == "completed"
        assert result.images[0].order_index == 0
        assert result.images[0].upload_id == "upload_aaa"
        assert result.images[0].translate_id == created.images[0].translate_id

    async def test_reflects_error_message(self, fake_redis: fakeredis.FakeRedis) -> None:
        request = BatchRequest(upload_ids=["upload_aaa"])
        created = await create_batch(request, URLS_1)

        await update_translate_status(created.images[0].translate_id, "failed", "GPU 메모리 부족")

        result = await get_batch(created.batch_id)

        assert result is not None
        assert result.images[0].status == "failed"
        assert result.images[0].error_message == "GPU 메모리 부족"

    async def test_missing_translate_treated_as_failed(
        self, fake_redis: fakeredis.FakeRedis
    ) -> None:
        request = BatchRequest(upload_ids=["upload_aaa"])
        created = await create_batch(request, URLS_1)

        fake_redis.delete(f"translate:{created.images[0].translate_id}")

        result = await get_batch(created.batch_id)

        assert result is not None
        assert result.status == "failed"
        assert result.images[0].status == "failed"
        assert result.images[0].error_message is not None
