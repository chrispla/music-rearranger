"""
Simple recording construction.
"""

import numpy as np  # for matrix processing


def construct_audio(y, sr, recon_beats, beat_times, crossfade):
    """
    crossfade in seconds
    """
    crossfade = int(sr*crossfade)
    # convert crossfade to even number if it's odd, so that the crossfade is symmetric
    if crossfade % 2 != 0:
        crossfade += 1

    y_new = np.array([])

    # generate general linear crossfade masks
    fade_in = [i / crossfade for i in range(crossfade)]
    fade_out = [1 - (i / crossfade) for i in range(crossfade)]

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

    # INTRO (only fade-out)
    start_frame = intervals_frames[0][0]
    end_frame = intervals_frames[0][1]
    # normal region
    y_new = np.concatenate((y_new, y[start_frame:end_frame-(crossfade//2)]))
    # fade-out region
    y_new = np.concatenate((y_new, y[end_frame-(crossfade//2):end_frame+(crossfade//2)]*fade_out))

    # MIDDLE INTERVALS
    for i in intervals_frames[1:-1]:
        start_frame = i[0]
        end_frame = i[1]
        # fade-in region
        y_new[-crossfade:] += y[start_frame-(crossfade//2):start_frame+(crossfade//2)]*fade_in
        # normal region
        y_new = np.concatenate((y_new, y[start_frame+(crossfade//2):end_frame-(crossfade//2)]))
        # fade-out region
        y_new = np.concatenate((y_new, y[end_frame-(crossfade//2):end_frame+(crossfade//2)]*fade_out))

    # OUTRO (only fade-in)
    start_frame = intervals_frames[-1][0]
    end_frame = intervals_frames[-1][1]
    # fade-in region
    y_new[-crossfade:] += y[start_frame-(crossfade//2):start_frame+(crossfade//2)]*fade_in
    y_new = np.concatenate((y_new, y[start_frame+(crossfade//2):end_frame]))

    return y_new
