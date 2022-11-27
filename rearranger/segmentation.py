"""
Encapsulating all the Adobe segmentation code in order to
get the needed data for the rearrangement.
"""

import numpy as np
import scipy

import musicsections


def stuff_I_need(
    filepath,
    deepsim_model_folder="models/deepsim",
    fewshot_model_folder="models/fewshot",
    n_levels=12,
    min_duration=8,
    mu=0.5,
    gamma=0.5,
    recsmooth=9,  # for segmentation, unfiltered will also be calculated
    recwidth=9,
    evecsmooth=9,
    normalize_matrices=True,
    distance="cosine",
    maxnorm=False
):

    model_deepsim = musicsections.load_deepsim_model(deepsim_model_folder)
    model_fewshot = musicsections.load_fewshot_model(fewshot_model_folder)

    # Compute beat-synced features
    Csync, Msync, Hsync, beat_times, audio_duration, beat_order = musicsections.core.make_beat_sync_features(
        filepath,
        deepsim_model=model_deepsim,
        fewshot_model=model_fewshot,
        beats_alg="beatnet",
        beats_file=None,
        use_mfcc=False,
        magicnorm=True
    )

    # Get filtered combined matrix
    A_f = musicsections.core.combined_matrix(
        Csync,
        Msync,
        Hsync,
        mu=mu,
        gamma=gamma,
        recsmooth=recsmooth,
        recwidth=recwidth,
        normalize_matrices=normalize_matrices,
        distance=distance,
        maxnorm=maxnorm
    )

    # Get unfiltered combined matrix
    A = musicsections.core.combined_matrix(
        Csync,
        Msync,
        Hsync,
        mu=mu,
        gamma=gamma,
        recsmooth=0,
        recwidth=recwidth,
        normalize_matrices=normalize_matrices,
        distance=distance,
        maxnorm=maxnorm
    )

    # Get Laplacian Eigenvectors
    L = scipy.sparse.csgraph.laplacian(A_f, normed=True)
    evals, evecs = scipy.linalg.eigh(L)
    embedding = scipy.ndimage.median_filter(evecs, size=(evecsmooth, 1))

    # Clustering
    # Cluster to obtain segmentations at k=1..n_levels levels
    Cnorm = np.cumsum(embedding**2, axis=1)**0.5
    segmentations = []
    for k in range(1, min(n_levels+1, embedding.shape[0]+1)):
        segmentations.append(musicsections.core.cluster(
            embedding, Cnorm, k, beat_times))

    # Reindex section IDs for multi-level consistency
    levels = musicsections.core.reindex(segmentations)

    fixed_levels = None
    if min_duration is None:
        fixed_levels = levels
    else:
        segs_list = []
        for i in range(1, len(levels) + 1):
            segs_list.append(musicsections.core.clean_segments(
                levels,
                min_duration=min_duration,
                fix_level=i,
                verbose=False))

    fixed_levels = musicsections.core.segments_to_levels(segs_list)

    return beat_times, beat_order, A, A_f, fixed_levels, Csync, Msync, Hsync
