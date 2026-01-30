import asyncio
import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import cast

from src.config import get_settings
from src.constants import TTL, Limits, RedisPrefix
from src.infra.redis import get_redis
from src.schemas.base import BaseSchema
from src.services.job_status import JobMetadata, JobStatus, TranslationOptions
from src.services.upload import get_upload
from src.tasks.process_job import process_job


class JobCreateRequest(BaseSchema):
    upload_ids: list[str]
    options: TranslationOptions


class JobResponse(BaseSchema):
    job_id: str
    status: JobStatus
    upload_ids: list[str]
    options: TranslationOptions
    created_at: str
    completed_at: str | None = None
    result_url: str | None = None
    error_message: str | None = None


class InvalidUploadError(Exception):
    def __init__(self, upload_id: str):
        self.upload_id = upload_id
        super().__init__(f"Invalid upload ID: {upload_id}")


class RateLimitExceededError(Exception):
    pass


def _generate_job_id() -> str:
    return f"job_{uuid.uuid4().hex[:8]}"


def _hash_ip(ip: str) -> str:
    secret = get_settings().ip_hash_secret
    return hashlib.sha256(f"{secret}:{ip}".encode()).hexdigest()[:16]


def _get_usage_key(client_ip: str) -> str:
    hashed_ip = _hash_ip(client_ip)
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return f"{RedisPrefix.USAGE}:{hashed_ip}:{today}"


async def _validate_upload_ids(upload_ids: list[str]) -> None:
    for upload_id in upload_ids:
        upload = await get_upload(upload_id)
        if upload is None:
            raise InvalidUploadError(upload_id)


_RATE_LIMIT_SCRIPT = """
local current = tonumber(redis.call("GET", KEYS[1]) or "0")
if current >= tonumber(ARGV[1]) then
    return -1
end
redis.call("INCR", KEYS[1])
redis.call("EXPIRE", KEYS[1], ARGV[2])
return current + 1
"""


async def _check_and_increment_usage(client_ip: str) -> None:
    redis = get_redis()
    usage_key = _get_usage_key(client_ip)

    result: int = redis.eval(  # type: ignore[assignment]
        _RATE_LIMIT_SCRIPT,
        1,
        usage_key,
        Limits.DAILY_JOB,
        TTL.USAGE,
    )

    if result == -1:
        raise RateLimitExceededError()


async def create_job(request: JobCreateRequest, client_ip: str) -> JobResponse:
    redis = get_redis()

    await _validate_upload_ids(request.upload_ids)
    await _check_and_increment_usage(client_ip)

    job_id = _generate_job_id()
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    metadata = JobMetadata(
        job_id=job_id,
        status="pending",
        upload_ids=request.upload_ids,
        options=request.options,
        created_at=created_at,
    )

    redis.set(f"{RedisPrefix.JOB}:{job_id}", metadata.model_dump_json(), ex=TTL.JOB)

    await asyncio.to_thread(process_job.delay, job_id)

    return JobResponse(
        job_id=metadata.job_id,
        status=metadata.status,
        upload_ids=metadata.upload_ids,
        options=metadata.options,
        created_at=metadata.created_at,
    )


async def get_job(job_id: str) -> JobResponse | None:
    redis = get_redis()

    data = redis.get(f"{RedisPrefix.JOB}:{job_id}")
    if data is None:
        return None

    metadata = JobMetadata.model_validate(json.loads(cast(str, data)))

    return JobResponse(
        job_id=metadata.job_id,
        status=metadata.status,
        upload_ids=metadata.upload_ids,
        options=metadata.options,
        created_at=metadata.created_at,
        completed_at=metadata.completed_at,
        result_url=metadata.result_url,
        error_message=metadata.error_message,
    )
