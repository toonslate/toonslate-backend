import fakeredis
import pytest

from src.constants import Limits
from src.services.quota import QuotaExceededError, check_and_consume_quota, hash_ip, refund_quota

HASHED_IP_A = hash_ip("127.0.0.1")
HASHED_IP_B = hash_ip("192.168.1.1")


class TestCheckAndConsumeQuota:
    async def test_first_usage_succeeds(self, fake_redis: fakeredis.FakeRedis) -> None:
        await check_and_consume_quota(HASHED_IP_A, 1)

    async def test_consume_multiple_images(self, fake_redis: fakeredis.FakeRedis) -> None:
        await check_and_consume_quota(HASHED_IP_A, 5)
        await check_and_consume_quota(HASHED_IP_A, 5)
        await check_and_consume_quota(HASHED_IP_A, 10)

    async def test_exceeds_weekly_limit(self, fake_redis: fakeredis.FakeRedis) -> None:
        await check_and_consume_quota(HASHED_IP_A, Limits.WEEKLY_IMAGES)

        with pytest.raises(QuotaExceededError):
            await check_and_consume_quota(HASHED_IP_A, 1)

    async def test_partial_exceed(self, fake_redis: fakeredis.FakeRedis) -> None:
        await check_and_consume_quota(HASHED_IP_A, 15)

        with pytest.raises(QuotaExceededError):
            await check_and_consume_quota(HASHED_IP_A, 6)

    async def test_different_ips_independent(self, fake_redis: fakeredis.FakeRedis) -> None:
        await check_and_consume_quota(HASHED_IP_A, Limits.WEEKLY_IMAGES)
        await check_and_consume_quota(HASHED_IP_B, Limits.WEEKLY_IMAGES)

    async def test_usage_key_has_ttl(self, fake_redis: fakeredis.FakeRedis) -> None:
        await check_and_consume_quota(HASHED_IP_A, 1)

        # fakeredis 타입 스텁이 decode_responses=True 반환 타입을 정확히 표현하지 못함
        keys = list(fake_redis.keys("usage:*"))  # type: ignore[reportUnknownMemberType]
        assert len(keys) == 1  # type: ignore[reportUnknownArgumentType]
        ttl = int(fake_redis.ttl(keys[0]))  # type: ignore[reportArgumentType]
        # 쿼터 TTL은 "다음 월요일까지 남은 초"로 동적 — 존재 여부만 검증
        assert ttl > 0


class TestRefundQuota:
    async def test_refund_decreases_count(self, fake_redis: fakeredis.FakeRedis) -> None:
        await check_and_consume_quota(HASHED_IP_A, 10)
        await refund_quota(HASHED_IP_A, 3)
        await check_and_consume_quota(HASHED_IP_A, 13)

    async def test_refund_does_not_go_below_zero(self, fake_redis: fakeredis.FakeRedis) -> None:
        await check_and_consume_quota(HASHED_IP_A, 2)
        await refund_quota(HASHED_IP_A, 5)
        await check_and_consume_quota(HASHED_IP_A, Limits.WEEKLY_IMAGES)


class TestCountValidation:
    async def test_consume_zero_raises(self, fake_redis: fakeredis.FakeRedis) -> None:
        with pytest.raises(ValueError):
            await check_and_consume_quota(HASHED_IP_A, 0)

    async def test_consume_negative_raises(self, fake_redis: fakeredis.FakeRedis) -> None:
        with pytest.raises(ValueError):
            await check_and_consume_quota(HASHED_IP_A, -1)

    async def test_refund_zero_raises(self, fake_redis: fakeredis.FakeRedis) -> None:
        with pytest.raises(ValueError):
            await refund_quota(HASHED_IP_A, 0)

    async def test_refund_negative_raises(self, fake_redis: fakeredis.FakeRedis) -> None:
        with pytest.raises(ValueError):
            await refund_quota(HASHED_IP_A, -1)
