from __future__ import annotations

from pathlib import Path

import joblib


def load_model_bundle(path: str | Path):
    """Load a saved model bundle from disk."""
    bundle = joblib.load(path)
    return bundle
