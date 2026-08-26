import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from methods.scgpt_lora.model import count_trainable, freeze_non_lora


class _MockParam:
    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad

    def numel(self):
        return 1


class _MockModel:
    """Simulates a model with LoRA and non-LoRA params."""
    def named_parameters(self):
        for name in ["lora_A", "lora_B", "head.weight", "transformer.weight"]:
            yield name, _MockParam("lora" in name)

    def parameters(self):
        yield _MockParam(True)
        yield _MockParam(True)
        yield _MockParam(False)


def test_count_trainable():
    m = _MockModel()
    assert count_trainable(m) == 2


def test_freeze_non_lora():
    m = _MockModel()
    freeze_non_lora(m)
    params = {name: p.requires_grad for name, p in m.named_parameters()}
    assert params["lora_A"] is True
    assert params["lora_B"] is True
    assert params["head.weight"] is False
