import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> int:
    """Set deterministic seeds across Python, NumPy, and environment."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed
