"""Path finding methods for transition positions.
"""

import copy


def get_transitions(points, n_beats, neighbors=True):

    import warnings
    warnings.warn("Using all transitions not implemented yet, returning ideal transitions...")

    return get_ideal_transitions(points, n_beats, neighbors=neighbors)


def get_ideal_transitions(points, n_beats, neighbors=True):
    transitions = {}
    # first possible candidate is neighboring beat
    if neighbors:
        for p in range(n_beats):
            transitions[p] = [p+1]
    for p in points:
        # for now only consider rank 1 points
        if p[1] == 1:
            if p[0][0] in transitions:
                if p[0][1] not in transitions[p[0][0]]:
                    transitions[p[0][0]].append(p[0][1])
            else:
                transitions[p[0][0]] = [p[0][1]]
    return transitions


def get_nonnegative_transitions(points, n_beats, neighbors=True):
    transitions = {}
    # first possible candidate is neighboring beat
    if neighbors:
        for p in range(n_beats):
            transitions[p] = [p+1]
    for p in points:
        # only forward transitions, any rank
        if p[0][1] > p[0][0]:
            if p[0][0] in transitions:
                if p[0][1] not in transitions[p[0][0]]:
                    transitions[p[0][0]].append(p[0][1])
            else:
                transitions[p[0][0]] = [p[0][1]]
    return transitions


def greedy_deep_search(cur_idx, transitions, rem_beats, n_beats, append="> "):

    beat_list = [cur_idx]

    if cur_idx == n_beats and rem_beats == 0:
        # Solution found!
        return beat_list
    elif cur_idx < n_beats and rem_beats > 0:
        targets = copy.deepcopy(transitions[cur_idx])
        for target_idx in targets:
            transitions[cur_idx].remove(target_idx)
            if target_idx != n_beats and cur_idx in transitions[target_idx]:
                transitions[target_idx].remove(cur_idx)

            result = greedy_deep_search(
                cur_idx=target_idx,
                transitions=copy.deepcopy(transitions),
                rem_beats=rem_beats-1,
                n_beats=n_beats,
                append=append+"> "
            )
            if result:
                return beat_list+result
        # If none returned, no path here
        return 0
    else:
        # Shouldn't be here?
        return 0


def less_transitions_algorithm(transitions, target_beats, total_beats, beat_analysis):
    """Simple, greedy algorithm that selects a path with the minimum
    number of transitions.
    """

    # get beats in measure
    beats_in_measure = int(max([b[1] for b in beat_analysis]))

    # get only points that are in the same position in the measure
    points = []
    for transition in transitions:
        if abs(transition[0][0]-transition[0][1]) % beats_in_measure == 0:
            points.append(transition[0])

    for beat_error in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7]:

        for point in points:
            path_size = point[0] + (total_beats-point[1])
            if path_size == target_beats + beat_error:
                beat_list = [p for p in range(point[0]+1)]
                beat_list += [p for p in range(point[1], total_beats+1)]
                return beat_list

        for point_1 in points:
            for point_2 in points:
                if point_2[0] < point_1[1]:
                    continue
                path_size = point_1[0] + (point_2[0]-point_1[1]) + (total_beats-point_2[1])
                if path_size == target_beats + beat_error:
                    beat_list = [p for p in range(point_1[0]+1)]
                    beat_list += [p for p in range(point_1[1], point_2[0]+1)]
                    beat_list += [p for p in range(point_2[1], total_beats+1)]
                    return beat_list

        for point_1 in points:
            for point_2 in points:
                if point_2[0] < point_1[1]:
                    continue
                for point_3 in points:
                    if point_3[0] < point_2[1]:
                        continue
                    path_size = point_1[0] + (point_2[0]-point_1[1])
                    path_size += (point_3[0]-point_2[1]) + (total_beats-point_3[1])
                    if path_size == target_beats + beat_error:
                        beat_list = [p for p in range(point_1[0]+1)]
                        beat_list += [p for p in range(point_1[1], point_2[0]+1)]
                        beat_list += [p for p in range(point_2[1], point_3[0]+1)]
                        beat_list += [p for p in range(point_3[1], total_beats+1)]
                        return beat_list

    print("No solutions found, exciting...")
    exit()
