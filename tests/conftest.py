import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from src.utils.storage.local import LocalStorage


@pytest.fixture
def temp_upload_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def local_storage(temp_upload_dir: Path) -> LocalStorage:
    return LocalStorage(base_dir=temp_upload_dir, base_url="/static")
