"""Where a trained LoRA adapter lives on the gvg-ckpts volume (mounted at /ckpts).

An adapter is named by its run, optionally with a checkpoint tag after "@":
    "prereg-grpo-s0"         -> /ckpts/prereg-grpo-s0/final   (the finished adapter)
    "prereg-grpo-s0@step10"  -> /ckpts/prereg-grpo-s0/step10  (an intermediate checkpoint)
Intermediate checkpoints are used by start K of the poisoning amendment
(docs/preregistration_poisoning_amendment_2026-11-26.md).
"""

from __future__ import annotations

import re


def adapter_dir(name: str, root: str = "/ckpts") -> str:
    """Folder of adapter `name` ("run" or "run@stepN") under `root`. Raises ValueError otherwise."""
    run, sep, tag = name.partition("@")
    if not run or "/" in run or (sep and not re.fullmatch(r"step\d+", tag)):
        raise ValueError(f"adapter must be 'run' or 'run@stepN', got {name!r}")
    return f"{root}/{run}/{tag if sep else 'final'}"
