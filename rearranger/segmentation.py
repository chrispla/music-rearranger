"""Segmentation methods.
"""

import os
import tempfile

import librosa
import numpy as np
import scipy
import sox
from BeatNet.BeatNet import BeatNet

import musicsections


def precise_segmentation(
    audio_filepath,
    config,
    deepsim_model_dir="models/deepsim",
    fewshot_model_dir="models/fewshot",
    use_gpu=False
):

    model_deepsim = musicsections.load_deepsim_model(deepsim_model_dir)
    model_fewshot = musicsections.load_fewshot_model(fewshot_model_dir, gpu=use_gpu)

    # Compute beat-synced features
    Csync, Msync, Hsync, beat_times, beat_analysis, _ = musicsections.core.make_beat_sync_features(
        filename=audio_filepath,
        deepsim_model=model_deepsim,
        fewshot_model=model_fewshot,
        beats_alg="beatnet",
        beats_file=None,
        use_mfcc=False,
        magicnorm=True)

    # Get filtered combined matrix
    R = musicsections.core.combined_matrix(
        Csync,
        Msync,
        Hsync,
        mu=config["mu"],
        gamma=config["gamma"],
        recsmooth=config["recsmooth"],
        recwidth=config["recwidth"],
        normalize_matrices=True,
        distance=config["distance"],
        maxnorm=False)

    # Get Laplacian Eigenvectors
    L = scipy.sparse.csgraph.laplacian(R, normed=True)
    _, evecs = scipy.linalg.eigh(L)
    embedding = scipy.ndimage.median_filter(evecs, size=(config["evecsmooth"], 1))

    # Clustering
    # Cluster to obtain segmentations at k=1..n_levels levels
    Cnorm = np.cumsum(embedding**2, axis=1)**0.5
    segmentations = []
    for k in range(1, min(config["n_levels"]+1, embedding.shape[0]+1)):
        segmentations.append(musicsections.core.cluster(
            embedding, Cnorm, k, beat_times))

    # Reindex section IDs for multi-level consistency
    levels = musicsections.core.reindex(segmentations)

    # Segment fusion
    segmentation = None
    if config["min_seg_size"] is None or config["min_seg_size"] == 0:
        segmentation = levels
    else:
        segs_list = []
        for i in range(1, len(levels) + 1):
            segs_list.append(musicsections.core.clean_segments(
                levels,
                min_duration=config["min_seg_size"],
                fix_level=i,
                verbose=False))

    segmentation = musicsections.core.segments_to_levels(segs_list)

    return segmentation, beat_times, beat_analysis, R, Csync, Msync, Hsync


def fast_segmentation(
    audio_filepath,
    config
):
    """Code modified from github.com/bmcfee/lsd_viz. See ./external/lsd_viz/ for license.
    """

    with tempfile.TemporaryDirectory() as tempdir:

        new_filepath = os.path.join(tempdir, os.path.basename(audio_filepath[:-4]) + "_22.wav")

        tfm = sox.Transformer()
        tfm.convert(samplerate=22050, n_channels=1)
        tfm.build(audio_filepath, new_filepath)

        # Load audio @ 22kHz as y
        y, sr = librosa.load(new_filepath, sr=None)

        # Compute Harmonic Constant-Q Transform in dB
        yh = librosa.effects.harmonic(y, margin=8)
        C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=yh,
                                                       sr=sr,
                                                       bins_per_octave=36,
                                                       n_bins=252,
                                                       hop_length=512)), ref=np.max)

        # Beat tracking
        estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
        beat_analysis = estimator.process(new_filepath)
        # get beat_times as formatted for the rest of the code
        beat_times = np.asarray([b[0] for b in beat_analysis])
        beats = librosa.time_to_frames(beat_times, sr=22050, hop_length=512)
        if beat_times[0] > 0:
            beat_times = np.insert(beat_times, 0, 0)
        if beat_times[-1] < len(y)/sr:
            beat_times = np.append(beat_times, len(y)/sr)

    # beat synchronize the CQT
    Csync = librosa.util.sync(C, beats, aggregate=np.median)

    # stack 4 consecutive frames
    Cstack = librosa.feature.stack_memory(Csync, n_steps=4)

    # compute weighted recurrence matrix
    R = librosa.segment.recurrence_matrix(Cstack,
                                          width=config["recwidth"],
                                          mode='affinity',
                                          sym=True)

    # enhance diagonals with a median filter
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, config["recsmooth"]))

    # compute MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    # beat synchronize them
    Msync = librosa.util.sync(mfcc, beats)

    # build the MFCC sequence matrix
    path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    # get the balanced combination of the MFCC sequence matric and the CQT
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)
    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)
    A = mu * Rf + (1 - mu) * R_path

    # compute the normalized Laplacian
    L = scipy.sparse.csgraph.laplacian(A, normed=True)

    # decompose
    _, evecs = scipy.linalg.eigh(L)

    embedding = scipy.ndimage.median_filter(evecs, size=(config["evecsmooth"], 1))

    # Clustering
    # Cluster to obtain segmentations at k=1..n_levels levels
    Cnorm = np.cumsum(embedding**2, axis=1)**0.5
    segmentations = []
    for k in range(1, min(config["n_levels"]+1, embedding.shape[0]+1)):
        segmentations.append(musicssctions.core.cluster(
            embedding, Cnorm, k, beat_times))

    # Reindex section IDs for multi-level consistency
    levels = musicsections.core.reindex(segmentations)

    # Segment fusion
    segmentation = None
    if config["min_seg_size"] is None:
        segmentation = levels
    else:
        segs_list = []
        for i in range(1, len(levels) + 1):
            segs_list.append(musicsections.core.clean_segments(
                levels,
                min_duration=config["min_seg_size"],
                fix_level=i,
                verbose=False))

    segmentation = musicsections.core.segments_to_levels(segs_list)

    # returning Csync twice here to retain the format of the precise function
    return segmentation, beat_times, beat_analysis, R, Csync, Msync, Csync
