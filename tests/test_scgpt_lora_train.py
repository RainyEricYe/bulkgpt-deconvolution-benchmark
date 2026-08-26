import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from methods.scgpt_lora.train import compute_metrics, normalize_proportions


def test_compute_metrics():
    pred = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
    true = np.array([[0.4, 0.4, 0.2], [0.2, 0.5, 0.3]])
    names = ["A", "B", "C"]
    result = compute_metrics(pred, true, names)
    assert "macro_avg" in result
    assert "A" in result
    assert 0 <= result["A"]["pearson_r"] <= 1.0
    assert result["macro_avg"]["rmse"] >= 0
