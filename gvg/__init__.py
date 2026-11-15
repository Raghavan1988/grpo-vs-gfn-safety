"""Shared library for the grpo-vs-gfn-safety project.

The modules in this package contain the reusable parts of the experiment.  Keeping
them separate from the Modal entry points makes the important logic testable on a
CPU before spending money on a GPU run.  Large libraries are imported inside the
functions that need them, so simple utilities remain easy to import.

What is a package?  A folder that contains a file called ``__init__.py`` is a Python
"package".  Python runs this file the first time anything does ``import gvg`` or
``from gvg import ...``.  Here the file holds only this description, so importing the
package is instant; the real code lives in the sibling modules:

    gvg/data.py            builds and hashes the frozen train / validation prompt sets
    gvg/prefill_attacks.py the "Sure, here is ..." style prefill phrases (train / held-out)
    gvg/wildguard.py       loads the WildGuard judge model and turns its answers into probabilities
    gvg/rewards.py         turns judge probabilities (and maths answers) into one reward per reply
    gvg/scoring.py         cheap rule-based scorers: keyword refusal, GSM8K answer, Wilson intervals
    gvg/rl.py              training loops: GRPO, trajectory balance (TB), rejection-sampling SFT
    gvg/hard_eval.py       harder evaluation suites

The cloud entry point that wires these together on GPUs is ``train/modal_train.py``.
"""
