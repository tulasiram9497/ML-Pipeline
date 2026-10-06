from __future__ import annotations

from pathlib import Path


def ensure_path_exists(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved
