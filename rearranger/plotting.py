"""Various plotting code.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from mir_eval import display


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
        start_beat = np.where(beat_times == seg[0])[0][0]
        end_beat = np.where(beat_times == seg[1])[0][0]
        # paint the segment border
        if seg != seg_level[-1]:
            segs[start_beat:end_beat, end_beat] = 255
            segs[end_beat, start_beat:end_beat] = 255
        segs[start_beat:end_beat, start_beat] = 255
        segs[start_beat, start_beat:end_beat] = 255

    plt.imshow(A[start_beat_plot:end_beat_plot, start_beat_plot:end_beat_plot])
    plt.imshow(segs[start_beat_plot:end_beat_plot, start_beat_plot:end_beat_plot], alpha=alpha)
    plt.title(f"Seg level {level} on recurrence matrix, beats [{start_beat_plot}:{end_beat_plot}]")


def save_useful_plots(
    output_dir,
    output_name,
    seg_method,
    segmentation,
    Csync,
    Msync,
    Hsync
):
    # create figure folder in output dir
    fig_dir = os.path.join(output_dir, "figures")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # plot segmentation (code modified from musicsections)
    inters = []
    labels = []
    for level in segmentation[::-1]:
        inters.append(level[0])
        labels.append(level[1])
    N = len(inters)
    fig, axs = plt.subplots(N, figsize=(20, 10))
    for level in range(N):
        display.segments(np.asarray(inters[level]), labels[level], ax=axs[level])
        axs[level].set_yticks([0.5])
        axs[level].set_yticklabels([N - level])
        axs[level].set_xticks([])
    axs[0].xaxis.tick_top()
    fig.subplots_adjust(top=0.8)  # Otherwise savefig cuts the top
    plt.title(f"{output_name} {seg_method} segmentation")
    plt.savefig(os.path.join(fig_dir, f"{output_name}_{seg_method}_segmentation.png"))

    if seg_method == "precise":
        # plot Csync
        plt.figure()
        plt.imshow(Csync)
        plt.title(f"{output_name} DeepSim features")
        plt.savefig(os.path.join(fig_dir, f"{output_name}_{seg_method}_DeepSim.png"))

        # plot Msync
        plt.figure()
        plt.imshow(Msync)
        plt.title(f"{output_name} Few-Shot features")
        plt.savefig(os.path.join(fig_dir, f"{output_name}_{seg_method}_Few-Shot.png"))

        # plot Hsync
        plt.figure()
        plt.imshow(Hsync)
        plt.title(f"{output_name} Harmonic CQT")
        plt.savefig(os.path.join(fig_dir, f"{output_name}_{seg_method}_CQT.png"))

    elif seg_method == "fast":
        # plot Msync
        plt.figure()
        plt.imshow(Msync)
        plt.title(f"{output_name} MFCCs")
        plt.savefig(os.path.join(fig_dir, f"{output_name}_{seg_method}_MFCCs.png"))

        # plot Hsync
        plt.figure()
        plt.imshow(Hsync)
        plt.title(f"{output_name} Harmonic CQT")
        plt.savefig(os.path.join(fig_dir, f"{output_name}_{seg_method}_CQT.png"))
