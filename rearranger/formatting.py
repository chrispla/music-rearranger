"""Methods for formatting and quantizing the segmentation.
"""

import numpy as np


def structure_time_to_beats(segmentation, beat_times):
    """
    Convert the segmentation from time to beats.
    """
    segmentation_beats = []
    for level in segmentation:
        current_level = ([], [])
        for boundaries, type in zip(level[0], level[1]):
            boundaries = (np.where(beat_times == boundaries[0])[0][0],
                          np.where(beat_times == boundaries[1])[0][0])
            current_level[0].append(boundaries)
            current_level[1].append(type)
        segmentation_beats.append(current_level)

    return segmentation_beats


def take_closest(downbeat_beats, beat):
    """
    Assign beat to closest beat that is downbeat.
    Agnostic to time signature. Proxy to quantization
    """
    pos = np.searchsorted(downbeat_beats, beat, side='left', sorter=None)
    if pos == 0:
        return downbeat_beats[0]
    # if larger than all, return last
    if pos == len(downbeat_beats):
        return downbeat_beats[-1]
    former = downbeat_beats[pos-1]
    latter = downbeat_beats[pos]
    # if equal distance from downbeats, return latter
    # interesting decision musicologically...
    if latter - beat <= beat - former:
        return latter
    else:
        return former


# might have some issues here with when the beat tracker detects the first beats
# for example sometimes it's adding a pickup measure where there's mostly silence
def quantize_to_measures(segmentation_beats, n_measures, beat_analysis, beat_times):
    """Quantize a beat-indexed structure as represented by Adobe to
    a minimum of n measures.
    """
    downbeat_times = np.asarray([b[0] for b in beat_analysis if b[1] == 1])
    n_measure_times = downbeat_times[::n_measures]

    # this is done informationally, not needed!
    # 0 and audio_length border - hacky, but intro entry and outro exit not considered anyway
    if downbeat_times[0] != 0:
        downbeat_times[0] = 0
    if downbeat_times[-1] != beat_times[-1]:
        downbeat_times = np.append(downbeat_times, beat_times[-1])
    downbeat_beats = [np.where(beat_times == t)[0][0] for t in downbeat_times]

    if n_measure_times[0] != 0:
        n_measure_times[0] = 0
    if n_measure_times[-1] != beat_times[-1]:
        n_measure_times = np.append(n_measure_times, beat_times[-1])
    n_measure_beats = [np.where(beat_times == t)[0][0] for t in n_measure_times]

    segmentation_n_measures = []
    for level in segmentation_beats:
        current_level = ([], [])
        for boundaries, type in zip(level[0], level[1]):
            # if only segment, take boundaries
            if len(level[0]) == 1:
                q_boundaries = (0, n_measure_beats[-1])
            # if first, always begin from 0
            elif boundaries == level[0][0]:
                q_boundaries = (0, take_closest(n_measure_beats, boundaries[1]))
            # if last, always end in fake end beat
            elif boundaries == level[0][-1]:
                q_boundaries = (take_closest(n_measure_beats, boundaries[0]),
                                n_measure_beats[-1])
            else:
                q_boundaries = (take_closest(n_measure_beats, boundaries[0]),
                                take_closest(n_measure_beats, boundaries[1]))
            current_level[0].append(q_boundaries)
            current_level[1].append(type)
        segmentation_n_measures.append(current_level)

    return segmentation_n_measures, downbeat_times, downbeat_beats, n_measure_beats


def get_unique_segments(segmentation):
    """
    Remove segments that have the exact same boundaries AND segment type
    to make optimization more efficient. Works on both time- and beat-based
    segmentation representations.
    """
    boundaries_type_pairs = []
    unique = []
    for k, level in enumerate(segmentation):
        # replicate original structure
        current_level = ([], [])
        for boundaries, segtype in zip(level[0], level[1]):
            if (boundaries, segtype, ) not in boundaries_type_pairs:
                boundaries_type_pairs.append((boundaries, segtype, ))
                current_level[0].append(boundaries)
                current_level[1].append(segtype)
        unique.append(current_level)

    return unique


def get_target_n_beats(target_seconds, beat_analysis):
    """Estimate how many beats are needed to create a rearrangement
    that is approximately the target_time.
    """
    mean_beat_duration = np.mean(np.diff(beat_analysis[:, 0]))
    target_n_beats = int(np.round(target_seconds / mean_beat_duration))

    return target_n_beats
