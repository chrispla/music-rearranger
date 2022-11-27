"""
Various plotting code.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_segmentation_on_A(
    A,
    level,
    beat_times,
    fixed_levels,
    alpha=0.2,
    start_beat_plot=0,
    end_beat_plot=-1
):
    """
    Given a recurrence matrix and a segmentation (in Adobe format),
    plot the segmentation on top of the recurrence matrix.
    """
    seg_level = fixed_levels[level][0]
    segs = np.zeros(A.shape)
    # draw lines at the beats dictated by the times above
    for seg in seg_level:
        start_beat = np.where(beat_times==seg[0])[0][0]
        end_beat = np.where(beat_times==seg[1])[0][0]
        # paint the segment border
        if seg != seg_level[-1]:
            segs[start_beat:end_beat, end_beat] = 255
            segs[end_beat, start_beat:end_beat] = 255
        segs[start_beat:end_beat, start_beat] = 255
        segs[start_beat, start_beat:end_beat] = 255

    plt.imshow(A[start_beat_plot:end_beat_plot, start_beat_plot:end_beat_plot])
    plt.imshow(segs[start_beat_plot:end_beat_plot, start_beat_plot:end_beat_plot], alpha=alpha)
    plt.title(f"Seg level {level} on recurrence matrix, beats [{start_beat_plot}:{end_beat_plot}]")
