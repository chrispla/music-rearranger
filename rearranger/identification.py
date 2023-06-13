"""
Using the deepsim, chroma, and fewshot recurrence matrices, identify
consecutive beat diagonals.
"""

import numpy as np
import librosa
import more_itertools


def long_diagonals(A,
                   length=4,  # consecutive beats over threshold in diagonal
                   percentile=90):
    """
    Given a recurrence matrix, find consecutive diagonals with
    defined minimum length. Optionally filter the recurrence
    matrix by getting points over a certain percentile.
    """
    # Get recurrence matrix (without sequence matrix, we'll deal with that later)
    A_rec = A.copy()
    for i in range(A_rec.shape[0]-1):
        A_rec[i+1, i] = 0
        A_rec[i, i+1] = 0

    # let's get the value of the threshold percentile
    connections = np.zeros(A.shape)
    threshold = np.percentile(A, percentile)

    # get all points over the threshold and keep their value
    for i in range(A_rec.shape[0]):
        for j in range(A_rec.shape[1]):
            if A_rec[i, j] > threshold:
                connections[i, j] = A_rec[i, j]

    # now let's mark diagonals that are more than 4 consecutive beats
    patterns = np.zeros(A_rec.shape)
    # traverse A_rec diagonally
    for i in range(A_rec.shape[0]):
        x = i
        y = 0
        while x < A_rec.shape[0]:
            # if it's a repetition beat
            if connections[x, y] != 0:
                # check if you can check the consecutive `length`
                # beats behind it in the diagonal
                if x > length and y > length:
                    # check if they are also repetition beats
                    consecutive_flag = True
                    for k in range(1, length):
                        if connections[x-k, y-k] == 0:
                            consecutive_flag = 0
                            break
                    # if they are consecutive, mark them in the
                    # repetition matrix
                    if consecutive_flag:
                        for k in range(length):
                            patterns[x-k, y-k] = 1
            x += 1
            y += 1

    return connections, patterns


def common_patterns(Csync, Msync, Hsync, length, percentile):
    """
    Given the beat-synced features, get a symmetric matrix of their
    common patterns, considering diagonals of a minimum length and
    points within the defined percentile.
    """

    # get patterns
    Crec = librosa.segment.recurrence_matrix(
        Csync,
        width=min(9, min(Csync.shape)),
        mode='affinity',
        metric="cosine",
        sym=True)

    Mrec = librosa.segment.recurrence_matrix(
        Msync,
        width=min(9, min(Msync.shape)),
        mode='affinity',
        metric="cosine",
        sym=True)

    Hrec = librosa.segment.recurrence_matrix(
        Hsync,
        width=min(9, min(Hsync.shape)),
        mode='affinity',
        metric="cosine",
        sym=True)

    Ccon, Cpat = long_diagonals(A=Crec, length=length, percentile=percentile)
    Mcon, Mpat = long_diagonals(A=Mrec, length=length, percentile=percentile)
    Hcon, Hpat = long_diagonals(A=Hrec, length=length, percentile=percentile)

    # and let's get the common ones
    all_pat = np.zeros(Cpat.shape)
    for i in range(Cpat.shape[0]):
        for j in range(Cpat.shape[1]):
            if Cpat[i, j] == 1 and Mpat[i, j] == 1 and Hpat[i, j] == 1:
                # weight point by similarity
                all_pat[i, j] = (Ccon[i, j] + Mcon[i, j] + Hcon[i, j])/3.

    # make symmetric of the lower triangle
    patterns = np.tril(all_pat) + np.triu(all_pat.T, 1)

    return patterns


def get_best_transition_point(current_boundaries,
                              candidate_boundaries,
                              radius,
                              patterns):
    """Get best neighbor beat transitions for 2 segments.
    Check for the longest, closest diagonal of at least 2 units.
    """
    # get matching beats
    points = []

    # move across the diagonal
    for r in range(-radius, min(radius+1,  # assumes 1st segment isn't candidate
                                patterns.shape[0]-candidate_boundaries[0],
                                patterns.shape[0]-current_boundaries[1])):
        if patterns[current_boundaries[1]+r, candidate_boundaries[0]+r] != 0:
            points.append(r)

    # get ranges of consecutive matches
    diagonals = [list(group) for group in more_itertools.consecutive_groups(points)]

    # if there are no diagonals, return boundary with a similarity of 0
    if not diagonals:
        return 0, 0

    # get longest, closest diagonal
    best_d = []
    for d in diagonals:
        if len(d) > len(best_d):
            best_d = d
        elif len(d) == len(best_d):
            # if same length, get closest
            if np.abs(np.mean(d)) < np.abs(np.mean(best_d)):
                best_d = d
    # hm, but is the boundary really better than a single point match?
    if len(best_d) < 2:
        return 0, 0

    # return the middle beat of best diagonal (round up if even), rank 1
    best_p = best_d[int(np.ceil(len(best_d)/2))]
    similarity = 1

    return best_p, similarity


def get_all_transition_points(current_boundaries,
                              candidate_boundaries,
                              radius,
                              patterns):
    """Get all neighbor beat transitions for 2 segments.
    Check for all diagonals of at least 2 consecutive
    pattern beats.
    """
    # get matching beats
    points = []

    # move across the diagonal
    for r in range(-radius, min(radius+1,  # assumes 1st segment isn't candidate
                                patterns.shape[0]-candidate_boundaries[0],
                                patterns.shape[0]-current_boundaries[1])):
        if patterns[current_boundaries[1]+r, candidate_boundaries[0]+r] != 0:
            points.append(r)

    # get ranges of consecutive matches
    diagonals = [list(group) for group in more_itertools.consecutive_groups(points)]

    # if there are no diagonals, return boundary with a similarity of 0
    if not diagonals:
        return [(0, 0)]

    all_points = []
    for d in diagonals:
        if len(d) < 2:
            # hm, but is the boundary really better than a single point match?
            return [(0, 0)]
        point = d[int(np.ceil(len(d)/2))]
        similarity = patterns[
            current_boundaries[1] + point,
            candidate_boundaries[0] + point]
        all_points.append((point, similarity))

    return all_points


def cross_segment_points(segmentation,
                         quantization,
                         beats_in_measure,
                         patterns,
                         point_types="all"):
    """Compute all entry/exit pairs for segment transitions. Uses the
    neighbor beat transition algorithm to find smoother points
    around segment boundaries.

    Args:
        segmentation (list): structure - make sure it only has unique segments,
                                         and is quantized as desired
        quantization (int): the number of measures the structure has been quantized to
        beats_in_measure (int): the number of beats in each measure
                                assuming it remains consistent (which is a bad
                                limitation of this rearranger :'))
        patterns (np.array): the common patterns matrix
        point_types (str): whether to return "all" neighbor points larger than min diagonal,
                           or only the "best" one for each segment transition
    Returns:
        segment_points (list): list of entry/exit pairs in the format
                               ([entry_beat, exit_beat], rank)])
    """

    segment_points = []
    count = 0

    # Iterate through all segments
    for level in segmentation:
        for current_boundaries, current_segtype in zip(level[0], level[1]):
            count += 1
            # Iterate through all candidate segments (first segment not included)
            for candidate_boundaries, candidate_segtype in zip(level[0][1:], level[1][1:]):

                # Discard candidate if same segment type
                if current_segtype == candidate_segtype:
                    continue
                # Discard candidate if overlapping
                if (candidate_boundaries[0] >= current_boundaries[0] and
                   candidate_boundaries[0] <= current_boundaries[1]):
                    continue
                if (candidate_boundaries[1] <= current_boundaries[1] and
                   candidate_boundaries[1] >= current_boundaries[0]):
                    continue
                if (candidate_boundaries[0] >= current_boundaries[0] and
                   candidate_boundaries[1] <= current_boundaries[1]):
                    continue
                if (candidate_boundaries[0] <= current_boundaries[0] and
                   candidate_boundaries[1] >= current_boundaries[1]):
                    continue

                # If suitable candidate, find best neighbor transition point(s)
                if point_types == "best":
                    best_point, similarity = get_best_transition_point(
                        current_boundaries=current_boundaries,
                        candidate_boundaries=candidate_boundaries,
                        radius=beats_in_measure*quantization,
                        patterns=patterns)
                    segment_points.append(
                        ([current_boundaries[1]+best_point, candidate_boundaries[0]+best_point],
                         similarity))
                if point_types == "all":
                    all_points = get_all_transition_points(
                        current_boundaries=current_boundaries,
                        candidate_boundaries=candidate_boundaries,
                        radius=beats_in_measure*quantization,
                        patterns=patterns)
                    for point, similarity in all_points:
                        segment_points.append(
                            ([current_boundaries[1]+point, candidate_boundaries[0]+point],
                             similarity))
    return segment_points


def intra_segment_points(segmentation,
                         levels_list,
                         min_d_len,
                         patterns,
                         beats_in_measure):
    """
    Compute all type transitions, meaning all within-segment
    transitions for the given levels_list that stem from a
    diagonal in the patterns matrix that is at least mid_d_len
    long (its middle point is chosen as the transition, with rank 1).
    """

    type_points = []

    for l in levels_list:
        for boundaries, segtype in zip(segmentation[l][0], segmentation[l][1]):
            # traverse diagonals which are across an integer multiple of
            # beats_in_measure, so that jumps only across beats with the
            # same "function" (i.e. a beat 3 with a beat 3) occur
            for i in range(0, boundaries[1]-boundaries[0], beats_in_measure):
                x = boundaries[0]
                y = boundaries[0] + i

                active_d = False
                entry_p = -1
                while y < boundaries[1]:
                    if patterns[x, y] != 0:
                        if active_d is False:
                            entry_p = [x, y]
                            active_d = True
                    else:  # if patterns[x, y] == 0
                        if active_d is True:
                            # if diagonal ends there, check if >=min_d_len
                            d_len = x - entry_p[0]
                            if d_len >= min_d_len:
                                # and save the mid point, if it doesn't already exist
                                mid_p = [
                                    entry_p[0]+int(np.ceil(d_len/2)),
                                    entry_p[1]+int(np.ceil(d_len/2))]
                                if (mid_p, 1) not in type_points:
                                    type_points.append((mid_p, 1))
                            active_d = False

                    x += 1
                    y += 1

    return type_points
