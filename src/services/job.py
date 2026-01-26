import json
import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from src.constants import TTL, Limits, RedisPrefix
from src.schemas.base import BaseSchema
from src.services.upload import get_upload
from src.tasks.process_job import process_job
from src.utils.redis import get_redis

JobStatus = Literal["pending", "processing", "completed", "failed"]


class TranslationOptions(BaseSchema):
    source_language: str
    target_language: str


class JobMetadata(BaseModel):
    job_id: str
    status: JobStatus
    upload_ids: list[str]
    options: TranslationOptions
    created_at: str
    completed_at: str | None = None
    result_url: str | None = None
    error_message: str | None = None


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


def _get_usage_key(client_ip: str) -> str:
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return f"{RedisPrefix.USAGE}:{client_ip}:{today}"


async def _validate_upload_ids(upload_ids: list[str]) -> None:
    for upload_id in upload_ids:
        upload = await get_upload(upload_id)
        if upload is None:
            raise InvalidUploadError(upload_id)


async def _check_rate_limit(client_ip: str) -> None:
    redis = get_redis()
    usage_key = _get_usage_key(client_ip)

    count = await redis.get(usage_key)
    if count is not None and int(count) >= Limits.DAILY_JOB:
        raise RateLimitExceededError()


async def _increment_usage(client_ip: str) -> None:
    redis = get_redis()
    usage_key = _get_usage_key(client_ip)

    pipe = redis.pipeline()
    pipe.incr(usage_key)
    pipe.expire(usage_key, TTL.USAGE)
    await pipe.execute()


async def create_job(request: JobCreateRequest, client_ip: str) -> JobResponse:
    redis = get_redis()

    await _validate_upload_ids(request.upload_ids)
    await _check_rate_limit(client_ip)

    job_id = _generate_job_id()
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    metadata = JobMetadata(
        job_id=job_id,
        status="pending",
        upload_ids=request.upload_ids,
        options=request.options,
        created_at=created_at,
    )

    await redis.set(f"{RedisPrefix.JOB}:{job_id}", metadata.model_dump_json(), ex=TTL.JOB)

    await _increment_usage(client_ip)

    process_job.delay(job_id)

    return JobResponse(
        job_id=metadata.job_id,
        status=metadata.status,
        upload_ids=metadata.upload_ids,
        options=metadata.options,
        created_at=metadata.created_at,
    )


async def get_job(job_id: str) -> JobResponse | None:
    redis = get_redis()

    data = await redis.get(f"{RedisPrefix.JOB}:{job_id}")
    if data is None:
        return None

    metadata = JobMetadata.model_validate(json.loads(data))

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


async def update_job_status(
    job_id: str,
    status: JobStatus,
    result_url: str | None = None,
    error_message: str | None = None,
) -> bool:
    redis = get_redis()

    data = await redis.get(f"{RedisPrefix.JOB}:{job_id}")
    if data is None:
        return False

    metadata = JobMetadata.model_validate(json.loads(data))
    metadata.status = status

    if status in ("completed", "failed"):
        metadata.completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    if result_url:
        metadata.result_url = result_url

    if error_message:
        metadata.error_message = error_message

    await redis.set(f"{RedisPrefix.JOB}:{job_id}", metadata.model_dump_json(), ex=TTL.JOB)
    return True
