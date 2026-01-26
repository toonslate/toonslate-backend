import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import UploadFile
from pydantic import BaseModel

from src.config import get_settings
from src.constants import TTL, RedisPrefix
from src.infra.redis import get_redis
from src.infra.storage import get_storage
from src.schemas.base import BaseSchema


class UploadMetadata(BaseModel):
    upload_id: str
    filename: str
    content_type: str
    size: int
    path: str
    created_at: str


class UploadResponse(BaseSchema):
    upload_id: str
    image_url: str
    filename: str
    content_type: str
    size: int
    created_at: str


def _generate_upload_id() -> str:
    return f"upload_{uuid.uuid4().hex[:8]}"


async def create_upload(file: UploadFile) -> UploadResponse:
    storage = get_storage()
    redis = get_redis()
    settings = get_settings()

    upload_id = _generate_upload_id()
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    path = await storage.save(file, subdir="original", filename=upload_id)

    metadata = UploadMetadata(
        upload_id=upload_id,
        filename=file.filename or "unknown",
        content_type=file.content_type or "application/octet-stream",
        size=file.size or 0,
        path=path,
        created_at=created_at,
    )

    try:
        await redis.set(
            f"{RedisPrefix.UPLOAD}:{upload_id}", metadata.model_dump_json(), ex=TTL.UPLOAD
        )
    except Exception:
        storage.delete(path)
        raise

    ext = Path(file.filename or "").suffix or ".jpg"
    image_url = f"{settings.base_url}/static/original/{upload_id}{ext}"

    return UploadResponse(
        upload_id=upload_id,
        image_url=image_url,
        filename=metadata.filename,
        content_type=metadata.content_type,
        size=metadata.size,
        created_at=metadata.created_at,
    )


async def get_upload(upload_id: str) -> UploadResponse | None:
    redis = get_redis()
    settings = get_settings()

    data = await redis.get(f"{RedisPrefix.UPLOAD}:{upload_id}")
    if data is None:
        return None

    metadata = UploadMetadata.model_validate(json.loads(data))
    ext = Path(metadata.path).suffix

    return UploadResponse(
        upload_id=metadata.upload_id,
        image_url=f"{settings.base_url}/static/original/{upload_id}{ext}",
        filename=metadata.filename,
        content_type=metadata.content_type,
        size=metadata.size,
        created_at=metadata.created_at,
    )
