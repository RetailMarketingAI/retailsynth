from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO_ROOT_DIR = Path(__file__).parent.parent.parent
try:
    dist_name = "retailsynth"
    __version__ = version(dist_name)
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
