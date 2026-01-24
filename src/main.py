from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.utils.storage import get_storage
from src.utils.storage.local import LocalStorage

app = FastAPI()

# LocalStorage인 경우에만 StaticFiles 마운트 (S3 전환 시 제거)
storage = get_storage()
if isinstance(storage, LocalStorage):
    app.mount("/static", StaticFiles(directory=storage.base_dir), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
