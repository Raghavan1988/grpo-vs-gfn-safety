"""Shared library for the grpo-vs-gfn-safety project.

The modules in this package contain the reusable parts of the experiment.  Keeping
them separate from the Modal entry points makes the important logic testable on a
CPU before spending money on a GPU run.  Large libraries are imported inside the
functions that need them, so simple utilities remain easy to import.
"""
