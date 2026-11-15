"""Project path helpers.

PROJECT_ROOT is the repository root. It is resolved from this file's location, so it is
correct when the package is installed in editable mode (``pip install -e .``).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path):
    """Return ``path`` unchanged if absolute, otherwise resolve it against PROJECT_ROOT."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
