import redis

from src.config import get_settings


class _RedisHolder:
    client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    if _RedisHolder.client is None:
        settings = get_settings()
        _RedisHolder.client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
    return _RedisHolder.client


def close_redis() -> None:
    if _RedisHolder.client is not None:
        _RedisHolder.client.close()
        _RedisHolder.client = None


def set_redis(client: redis.Redis | None) -> None:
    _RedisHolder.client = client
