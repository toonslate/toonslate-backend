import json
from datetime import UTC, datetime
from typing import Literal, cast

from pydantic import BaseModel

from src.constants import TTL, RedisPrefix
from src.infra.redis import get_redis
from src.schemas.base import BaseSchema

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


def update_job_status(
    job_id: str,
    status: JobStatus,
    result_url: str | None = None,
    error_message: str | None = None,
) -> bool:
    redis = get_redis()

    data = redis.get(f"{RedisPrefix.JOB}:{job_id}")
    if data is None:
        return False

    metadata = JobMetadata.model_validate(json.loads(cast(str, data)))
    metadata.status = status

    if status in ("completed", "failed"):
        metadata.completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    if result_url:
        metadata.result_url = result_url

    if error_message:
        metadata.error_message = error_message

    redis.set(f"{RedisPrefix.JOB}:{job_id}", metadata.model_dump_json(), ex=TTL.JOB)
    return True
