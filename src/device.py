"""Choose the torch device the models run on."""

import sys


def pick_device(preference: str) -> str:
    """Resolve `--device auto` to cuda, mps or cpu, whichever is available."""
    if preference != "auto":
        return preference
    try:
        import torch
    except ImportError as err:
        print(f"! torch is not importable ({err}); using CPU", file=sys.stderr)
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_device(name: str) -> str:
    """Translate a CLI device name into one torch accepts.

    Ultralytics takes a bare cuda index, torch wants `cuda:0`.
    """
    return f"cuda:{name}" if name.isdigit() else name
