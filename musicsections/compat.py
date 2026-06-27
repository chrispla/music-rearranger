"""Compatibility helpers for older audio dependencies."""

import collections
import collections.abc
import itertools as it
import warnings

import numpy as np


def patch_collections_abc():
    """Expose moved ABC names expected by madmom on modern Python."""
    for name in [
        "Callable",
        "Iterable",
        "Mapping",
        "MutableMapping",
        "MutableSequence",
        "MutableSet",
        "Sequence",
    ]:
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))


def patch_numpy_aliases():
    """Expose removed NumPy aliases expected by madmom."""
    aliases = {
        "bool": bool,
        "complex": complex,
        "float": float,
        "int": int,
        "object": object,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def patch_madmom_downbeats():
    """Patch madmom's DBN downbeat processor for modern NumPy."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pkg_resources is deprecated as an API.*",
                category=UserWarning,
            )
            from madmom.features import downbeats
    except ImportError:
        return

    current_process = downbeats.DBNDownBeatTrackingProcessor.process
    if getattr(current_process, "_music_rearranger_patched", False):
        return

    def process(self, activations, **kwargs):
        first = 0
        if self.threshold:
            idx = np.nonzero(activations >= self.threshold)[0]
            if idx.any():
                first = max(first, np.min(idx))
                last = min(len(activations), np.max(idx) + 1)
            else:
                last = first
            activations = activations[first:last]
        if not activations.any():
            return np.empty((0, 2))

        results = list(self.map(
            downbeats._process_dbn,
            zip(self.hmms, it.repeat(activations))))
        best = int(np.argmax([result[1] for result in results]))
        path, _ = results[best]

        st = self.hmms[best].transition_model.state_space
        om = self.hmms[best].observation_model
        positions = st.state_positions[path]
        beat_numbers = positions.astype(int) + 1
        if self.correct:
            beats = np.empty(0, dtype=np.int)
            beat_range = om.pointers[path] >= 1
            idx = np.nonzero(np.diff(beat_range.astype(np.int)))[0] + 1
            if beat_range[0]:
                idx = np.r_[0, idx]
            if beat_range[-1]:
                idx = np.r_[idx, beat_range.size]
            if idx.any():
                for left, right in idx.reshape((-1, 2)):
                    peak = np.argmax(activations[left:right]) // 2 + left
                    beats = np.hstack((beats, peak))
        else:
            beats = np.nonzero(np.diff(beat_numbers))[0] + 1

        return np.vstack(((beats + first) / float(self.fps),
                          beat_numbers[beats])).T

    process._music_rearranger_patched = True
    downbeats.DBNDownBeatTrackingProcessor.process = process


def patch_legacy_audio_dependencies():
    patch_collections_abc()
    patch_numpy_aliases()
    patch_madmom_downbeats()
