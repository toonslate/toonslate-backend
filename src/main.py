from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.routes.upload import router as upload_router
from src.utils.redis import close_redis
from src.utils.storage import get_storage
from src.utils.storage.local import LocalStorage


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await close_redis()


app = FastAPI(lifespan=lifespan)
app.include_router(upload_router)

# LocalStorage인 경우에만 StaticFiles 마운트 (S3 전환 시 제거)
storage = get_storage()
if isinstance(storage, LocalStorage):
    storage.base_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=storage.base_dir), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
