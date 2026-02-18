"""주간 이미지 쿼터 관리

단일 번역(/translate)과 배치(/batch) 모두 같은 쿼터를 공유한다.
IP 기준, ISO week 단위, 이미지 수로 차감.
"""

import hashlib
from datetime import UTC, datetime, timedelta

from src.config import get_settings
from src.constants import Limits, RedisPrefix
from src.infra.redis import get_redis


class QuotaExceededError(Exception):
    pass


def hash_ip(ip: str) -> str:
    secret = get_settings().ip_hash_secret
    return hashlib.sha256(f"{secret}:{ip}".encode()).hexdigest()[:16]


def _get_quota_key(hashed_ip: str) -> str:
    now = datetime.now(UTC)
    iso_year, iso_week, _ = now.isocalendar()
    return f"{RedisPrefix.USAGE}:images:{hashed_ip}:{iso_year}-W{iso_week:02d}"


def _seconds_until_next_monday() -> int:
    now = datetime.now(UTC)
    days_ahead = 7 - now.weekday()  # Monday = 0
    next_monday = (now + timedelta(days=days_ahead)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return max(int((next_monday - now).total_seconds()), 1)


_CONSUME_SCRIPT = """
local current = tonumber(redis.call("GET", KEYS[1]) or "0")
local requested = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
if current + requested > limit then
    return -1
end
redis.call("INCRBY", KEYS[1], requested)
redis.call("EXPIRE", KEYS[1], ARGV[3])
return current + requested
"""

_REFUND_SCRIPT = """
local current = tonumber(redis.call("GET", KEYS[1]) or "0")
local refund = tonumber(ARGV[1])
local new_val = current - refund
if new_val < 0 then
    new_val = 0
end
redis.call("SET", KEYS[1], new_val, "KEEPTTL")
return new_val
"""


async def check_and_consume_quota(hashed_ip: str, count: int) -> None:
    """쿼터 차감. 초과 시 QuotaExceededError 발생."""
    if count <= 0:
        raise ValueError(f"count는 양수여야 합니다: {count}")
    redis = get_redis()
    key = _get_quota_key(hashed_ip)
    ttl = _seconds_until_next_monday()

    result: int = redis.eval(  # type: ignore[assignment]
        _CONSUME_SCRIPT,
        1,
        key,
        count,
        Limits.WEEKLY_IMAGES,
        ttl,
    )

    if result == -1:
        raise QuotaExceededError()


async def refund_quota(hashed_ip: str, count: int) -> None:
    """쿼터 환급 (큐잉 실패 시). 0 이하로 내려가지 않는다."""
    if count <= 0:
        raise ValueError(f"count는 양수여야 합니다: {count}")
    redis = get_redis()
    key = _get_quota_key(hashed_ip)

    redis.eval(  # type: ignore[union-attr]
        _REFUND_SCRIPT,
        1,
        key,
        count,
    )
