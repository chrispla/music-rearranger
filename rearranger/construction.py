"""
Simple recording construction.
"""

import numpy as np  # for matrix processing


def construct_audio(y, sr, recon_beats, beat_times, crossfade):
    """
    crossfade in seconds
    """
    crossfade = int(sr*crossfade)

    # from beat encoding to intervals
    intervals = []
    start_beat = recon_beats[0]
    for i in range(1, len(recon_beats)):
        # if they aren't consecutive
        if recon_beats[i-1] != recon_beats[i]-1:
            intervals.append([start_beat, recon_beats[i-1]])
            start_beat = recon_beats[i]
    intervals.append([start_beat, recon_beats[-1]])

    # convert beats to frames for given sr
    intervals_frames = []
    for interval in intervals:
        intervals_frames.append([int(beat_times[interval[0]]*sr),
                                 int(beat_times[interval[1]]*sr)])

    y_new = np.array([])
    for start_frame, end_frame in intervals_frames:
        if end_frame <= start_frame:
            continue

        segment = y[start_frame:end_frame]
        if y_new.size == 0:
            y_new = segment.copy()
            continue

        overlap = min(crossfade, len(y_new), len(segment))
        if overlap == 0:
            y_new = np.concatenate((y_new, segment))
            continue

        fade_in = np.linspace(0, 1, overlap, endpoint=False)
        fade_out = 1 - fade_in
        y_new[-overlap:] = y_new[-overlap:] * fade_out + segment[:overlap] * fade_in
        y_new = np.concatenate((y_new, segment[overlap:]))

    return y_new
