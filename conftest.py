import sys
from pathlib import Path
from warnings import warn

import pytest

BASE_DIR = Path(__file__).parent


def iter_parents(path: Path):
    if path.is_dir():
        yield path
    while path != path.root:
        path = path.parent
        yield path


path = Path.cwd()
for path in iter_parents(Path.cwd()):
    if (path / "supermat").is_dir():
        str_path = str(path)
        if str_path not in sys.path:
            sys.path.insert(0, str_path)
        break
else:
    warn("Could not find a parent directory which contained the supermat module.")


@pytest.fixture(scope="session")
def test_json() -> Path:
    return BASE_DIR / "test_samples/test.json"
