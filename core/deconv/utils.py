import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_project_root() -> Path:
    """Auto-detect project root by traversing up from ``core/deconv/utils.py``."""
    return Path(__file__).resolve().parent.parent.parent


def setup_logging(log_path: str) -> tuple[Path, object, object]:
    """Create log directory and return ``(log_path, tee_fn, file_handle)``.

    The *tee_fn* writes timestamped messages to both stdout and the log file.
    Caller is responsible for closing the file handle.
    """
    lp = Path(log_path)
    lp.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(lp, "a", buffering=1)

    def tee(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}\n"
        sys.stdout.write(line)
        sys.stdout.flush()
        log_fh.write(line)
        log_fh.flush()

    return lp, tee, log_fh


def external_dir() -> Path:
    """Return the path to ``data/external/`` (pinned external dependencies)."""
    return find_project_root() / "data" / "external"


def renormalize_props(arr: np.ndarray, zero_fill: str = "uniform") -> np.ndarray:
    """Clip negative values and renormalize each row to sum to 1.

    Parameters
    ----------
    arr:
        2-D array of shape (n_samples, n_types).
    zero_fill:
        ``"uniform"``: rows summing to 0 become equal proportions (1/n_types).
        ``"zero"``: rows summing to 0 stay at zero.
    """
    arr = np.maximum(arr, 0.0)
    row_sums = arr.sum(axis=1, keepdims=True)
    if zero_fill == "uniform":
        n_types = arr.shape[1]
        uniform_row = np.ones((1, n_types), dtype=arr.dtype) / n_types
        safe_sums = np.where(row_sums == 0, 1.0, row_sums)
        arr = arr / safe_sums
        arr[row_sums.squeeze(1) == 0] = uniform_row
    else:
        row_sums[row_sums == 0] = 1.0
        arr = arr / row_sums
    return arr
