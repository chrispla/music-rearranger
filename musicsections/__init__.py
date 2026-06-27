#!/usr/bin/env python
"""Top-level module for musicsections"""

from .version import version as __version__


def load_deepsim_model(*args, **kwargs):
    from .deepsim import load_deepsim_model as _load_deepsim_model
    return _load_deepsim_model(*args, **kwargs)


def load_fewshot_model(*args, **kwargs):
    from .fewshot import load_fewshot_model as _load_fewshot_model
    return _load_fewshot_model(*args, **kwargs)


def segment_file(*args, **kwargs):
    from .core import segment_file as _segment_file
    return _segment_file(*args, **kwargs)
