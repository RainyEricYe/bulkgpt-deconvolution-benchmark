import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from methods.scgpt_lora.data import normalize_proportions, load_h5_bulk


def test_normalize_proportions():
    pred = np.array([[0.5, -0.2, 1.0], [0.0, 0.1, 0.0]])
    normed = normalize_proportions(pred)
    assert normed.shape == (2, 3)
    assert np.allclose(normed.sum(axis=1), 1.0)
    assert np.all(normed >= 0)
    assert np.allclose(normed[0], [0.3333, 0.0, 0.6667], atol=1e-3)
