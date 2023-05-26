"""End-to-end calls for music rearrangement.
"""

import argparse
import os
import pickle

import librosa
import numpy as np
import soundfile as sf
import yaml

from rearranger.construction import construct_audio
from rearranger.formatting import (get_target_n_beats, get_unique_segments,
                                   quantize_to_measures,
                                   structure_time_to_beats)
from rearranger.identification import (common_patterns, cross_segment_points,
                                       intra_segment_points)
from rearranger.optimization import (get_ideal_transitions, get_transitions,
                                     get_nonnegative_transitions,
                                     less_transitions_algorithm, greedy_deep_search)
from rearranger.plotting import save_useful_plots
from rearranger.segmentation import fast_segmentation, precise_segmentation

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Music rearranger.")
    parser.add_argument(
        "--input_audio",
        required=True,
        type=str,
        help="input audio file path")
    parser.add_argument(
        "--target_time",
        required=True,
        type=int,
        help="target length of the rearranged audio in seconds")
    parser.add_argument(
        "--input_seg",
        required=False,
        type=str,
        help="""input segmentation information file path. Specify if you
                don't want to resegment the audio.""")
    parser.add_argument(
        "--output_dir",
        default="./output/",
        type=str,
        help="output directory for audio, segmentation information, and plots")
    parser.add_argument(
        "--seg_method",
        required=False,
        type=str,
        default="precise",
        help="""segmentation method to use. Options are:
                -'precise': uses the Salamon et al. 2021 method, requires
                             a large GPU, otherwise very slow on CPU.
                -'fast': uses the McFee & Ellis 2014 method on CPU.""")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="path to config file with various segmentation and rearrangement parameters")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="plot key elements of the rearrangement process")
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="use GPU for segmentation. Only applicable for 'precise' segmentation.")
    args = parser.parse_args()

    input_audio_path = args.input_audio
    target_seconds = args.target_time
    input_seg_path = args.input_seg
    output_dir = args.output_dir
    seg_method = args.seg_method
    config_path = args.config
    plot = args.plot
    use_gpu = args.use_gpu
    if not use_gpu:
        use_gpu = False

    # yaml load config file
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # segment if segmentation information file not provided
    if input_seg_path is None:
        print("> Segmenting...")
        if seg_method == "precise":
            segmentation, beat_times, beat_analysis, R, Csync, Msync, Hsync = precise_segmentation(
                audio_filepath=input_audio_path,
                config=config,
                deepsim_model_dir="models/deepsim",
                fewshot_model_dir="models/fewshot",
                use_gpu=use_gpu)
        elif seg_method == "fast":
            import warnings
            warnings.warn("Using fast segmentation. Point similarity and segmentation will be "
                          "based on CQT and MFCC similarity, which may not be very accurate.",
                          stacklevel=2)
            segmentation, beat_times, beat_analysis, R, Csync, Msync, Hsync = fast_segmentation(
                audio_filepath=input_audio_path,
                config=config)
        else:
            raise ValueError("Invalid segmentation method.")

        # Write segmentation information to file
        if output_dir is None:
            output_seg_path = input_audio_path[:-4] + f"_{seg_method}.pkl"
        else:
            output_seg_path = os.path.join(
                output_dir,
                os.path.basename(input_audio_path[:-4]) + f"_{seg_method}.pkl")
        with open(output_seg_path, "wb") as f:
            pickle.dump([segmentation, beat_times, beat_analysis, R, Csync, Msync, Hsync], f)
    else:
        print("> Loading segmentation information...")
        # load already computed segmentation information
        with open(input_seg_path, "rb") as f:
            segmentation, beat_times, beat_analysis, R, Csync, Msync, Hsync = pickle.load(f)

    # Save useful segmentation-related plots
    print("> Saving segmentation plots...")
    if plot:
        save_useful_plots(
            output_dir=output_dir,
            output_name=os.path.basename(input_audio_path[:-4]),
            seg_method=seg_method,
            segmentation=segmentation,
            Csync=Csync,
            Msync=Msync,
            Hsync=Hsync)

    # Format and quantize structure representation
    segmentation_beats = structure_time_to_beats(
        segmentation=segmentation,
        beat_times=beat_times)
    segmentation_n_measures, downbeat_times, downbeat_beats, n_measure_beats = quantize_to_measures(
        segmentation_beats=segmentation_beats,
        n_measures=config["min_measure"],
        beat_analysis=beat_analysis,
        beat_times=beat_times)
    segmentation_n_measures_unique = get_unique_segments(segmentation_beats)

    print("> Identifying transition points...")
    # Get patterns
    patterns = common_patterns(
        Csync=Csync,
        Msync=Msync,
        Hsync=Hsync,
        length=config["cross_pattern_length"],
        percentile=config["cross_pattern_percentile"])

    # Get cross-segment points
    cross_points = cross_segment_points(
        segmentation=segmentation_n_measures_unique,
        quantization=config["min_measure"],
        beats_in_measure=int(np.max(beat_analysis[:, 1])),
        patterns=patterns)
    print("  > Cross-segment points:", len(cross_points))
    print("    > Rank 1 points:", len([p for p in cross_points if p[1] == 1]))
    print("    > Rank 2 points:", len([p for p in cross_points if p[1] == 2]))

    # Get intra-segment points
    intra_points = intra_segment_points(
        segmentation=segmentation_n_measures,
        levels_list=config["intra_levels_list"],
        min_d_len=config["intra_pattern_length"],
        patterns=patterns,
        beats_in_measure=int(np.max(beat_analysis[:, 1])))
    print("  > Intra-segment points:", len(intra_points))

    print("> Finding rearrangement path...")
    # Path finding
    if config["transition_types"] == "ideal":
        transitions = get_ideal_transitions(
            cross_points+intra_points,
            patterns.shape[0],
            neighbors=True)
        jumps = [p for p in intra_points+cross_points if p[1] == 1]
    elif config["transition_types"] == "nonnegative":
        transitions = get_nonnegative_transitions(
            cross_points+intra_points,
            patterns.shape[0],
            neighbors=True)
        jumps = [p for p in intra_points+cross_points if p[1] > p[0]]
    elif config["transition_types"] == "all":
        transitions = get_transitions(
            cross_points+intra_points,
            patterns.shape[0],
            neighbors=True)
        jumps = intra_points + cross_points
    else:
        raise ValueError("Invalid transition type.")

    if config["path_finding_algorithm"] == "greedy_deep_search":
        import warnings
        warnings.warn("Greedy deep search algorithm is unstable and might hang.",
                      stacklevel=2)
        beat_list = greedy_deep_search(
            cur_idx=0,
            transitions=transitions,
            rem_beats=get_target_n_beats(target_seconds, beat_analysis),
            n_beats=patterns.shape[0])
    elif config["path_finding_algorithm"] == "less_transitions":
        import warnings
        warnings.warn("""Searching for rearrangement with up to 3 transitions. The solution
                         might not be found, or might not be optimal.""",
                      stacklevel=2)
    beat_list = less_transitions_algorithm(
        transitions=jumps,
        target_beats=get_target_n_beats(target_seconds, beat_analysis),
        total_beats=patterns.shape[0],
        beat_analysis=beat_analysis)

    print("> Saving rearrangement...")
    # Construct audio
    y, sr = librosa.load(path=input_audio_path, sr=None)
    y_rearranged = construct_audio(
        y=y,
        sr=sr,
        recon_beats=beat_list,
        beat_times=beat_times,
        crossfade=0.1)

    # Write audio
    if output_dir is None:
        output_audio_path = input_audio_path[:-4] + "_" + str(target_seconds) + "s.wav"
    else:
        output_audio_path = os.path.join(
            output_dir,
            os.path.basename(input_audio_path[:-4]) + "_" + str(target_seconds) + "s.wav")
    sf.write(output_audio_path, y_rearranged, sr)

    print("Done!")
