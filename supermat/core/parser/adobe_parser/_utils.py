import os
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Self, TypedDict

import orjson
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

TMP_DIR = os.environ.get("TMP_DIR", None)


class CacheIndex(TypedDict):
    timestamp: float
    cached_file_path: Path


def get_persistent_temp_directory(app_name="adobe_cache") -> Path:
    base_temp_dir = Path(TMP_DIR or tempfile.gettempdir())
    persistent_dir = base_temp_dir / f"{app_name}_temp"

    # Create the directory if it doesn't exist
    persistent_dir.mkdir(parents=True, exist_ok=True)

    return persistent_dir


def orjson_defaults(obj):
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError


class CachedFile:
    _instance: Self | None = None
    _lock = threading.Lock()
    _cache_dir: Path | None = None
    _cache_index: dict[Path, CacheIndex] = {}
    _cache_index_filename = "__cache_index.json"
    max_cache_size = 100

    def __new__(cls):
        CachedFile._setup_tmp_dir()
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def _setup_tmp_dir():
        if CachedFile._cache_dir is None:
            CachedFile._cache_dir = get_persistent_temp_directory()
            CachedFile.reload_cache_index()

    def __init__(self):
        assert CachedFile._cache_dir is not None
        self._cache_path = CachedFile._cache_dir

    def exists(self, pdf_file: Path) -> bool:
        return pdf_file in CachedFile._cache_index

    def get_cached_file_path(self, pdf_file: Path) -> Path:
        return CachedFile._cache_index[pdf_file]["cached_file_path"]

    @staticmethod
    def get_cache_index_path() -> Path | None:
        if CachedFile._cache_dir is None:
            return None
        return Path(CachedFile._cache_dir) / CachedFile._cache_index_filename

    @staticmethod
    def update_cache_file():
        cache_index_path = CachedFile.get_cache_index_path()
        if cache_index_path is None:
            return

        with cache_index_path.open("wb+") as fp:
            fp.write(
                orjson.dumps({k.as_posix(): v for k, v in CachedFile._cache_index.items()}, default=orjson_defaults)
            )

    @staticmethod
    def reload_cache_index():
        cache_index_path = CachedFile.get_cache_index_path()
        if cache_index_path is not None and cache_index_path.exists():
            with cache_index_path.open("rb") as fp:
                CachedFile._cache_index = {
                    Path(path): CacheIndex(**cache_index) for path, cache_index in orjson.loads(fp.read()).items()
                }

    def create_file(self, pdf_file: Path, suffix: str = ".zip") -> Path:
        cache_index = CacheIndex(
            timestamp=datetime.now().timestamp(),
            cached_file_path=self._cache_path / pdf_file.with_suffix(pdf_file.suffix + suffix).name,
        )
        CachedFile._cache_index[pdf_file] = cache_index
        CachedFile.cleanup_cache()
        CachedFile.update_cache_file()
        return self.get_cached_file_path(pdf_file)

    @staticmethod
    def cleanup_cache():
        if len(CachedFile._cache_index) > CachedFile.max_cache_size:
            oldest = sorted(CachedFile._cache_index.items(), key=lambda x: x[1]["timestamp"])[0]
            del_cache_index = CachedFile._cache_index.pop(oldest[0])
            del_cache_index["cached_file_path"].unlink(missing_ok=True)

    @staticmethod
    def clear_index():
        CachedFile._cache_index.clear()

    @staticmethod
    def cleanup():
        if CachedFile._cache_dir:
            shutil.rmtree(CachedFile._cache_dir)
            CachedFile._cache_dir = get_persistent_temp_directory()


CachedFile()
