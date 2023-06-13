"""Path finding methods for transition positions.
"""

import copy
from joblib import Parallel, delayed


def get_transitions(points, n_beats, type="ideal", neighbors=True):

    transitions = {}
    similarities = {}
    # first possible candidate is neighboring beat
    if neighbors:
        for p in range(n_beats):
            transitions[p] = [p+1]
    for p in points:
        if type == "ideal":
            if p[1] == 0:
                continue
        elif type == "nonnegative":
            if p[0][1] < p[0][0]:
                continue
        elif type != "all":
            raise ValueError("Invalid transition type.")
        if p[0][0] in transitions:
            if p[0][1] not in transitions[p[0][0]]:
                transitions[p[0][0]].append(p[0][1])
                similarities[(p[0][0], p[0][1])] = p[1]
        else:
            transitions[p[0][0]] = [p[0][1]]
            similarities[(p[0][0], p[0][1])] = p[1]
    return transitions, similarities


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


def paths_with_up_to_3_transitions(
    points,
    similarities,
    target_beats,
    total_beats
):
    # we'll keep track of transition similarities, reguralized by transitions number

    solutions = []
    for point in points:
        reg_sim = similarities[point[0], point[1]]
        path_size = point[0] + (total_beats-point[1])
        if path_size == target_beats:
            beat_list = [p for p in range(point[0]+1)]
            beat_list += [p for p in range(point[1], total_beats+1)]
            solutions.append((beat_list, reg_sim))

    for point_1 in points:
        for point_2 in points:
            reg_sim = (similarities[point[0], point[1]] +
                       similarities[point_1[0], point_1[1]]) / 2.
            if point_2[0] < point_1[1]:
                continue
            reg_sim = similarities[point[0], point[1]]
            path_size = point_1[0] + (point_2[0]-point_1[1]) + (total_beats-point_2[1])
            if path_size == target_beats:
                beat_list = [p for p in range(point_1[0]+1)]
                beat_list += [p for p in range(point_1[1], point_2[0]+1)]
                beat_list += [p for p in range(point_2[1], total_beats+1)]
                solutions.append((beat_list, reg_sim))

    for point_1 in points:
        for point_2 in points:
            if point_2[0] < point_1[1]:
                continue
            for point_3 in points:
                if point_3[0] < point_2[1]:
                    continue
                reg_sim = (similarities[point[0], point[1]] +
                           similarities[point_1[0], point_1[1]] +
                           similarities[point_2[0], point_2[1]]) / 3.
                path_size = point_1[0] + (point_2[0]-point_1[1])
                path_size += (point_3[0]-point_2[1]) + (total_beats-point_3[1])
                if path_size == target_beats:
                    beat_list = [p for p in range(point_1[0]+1)]
                    beat_list += [p for p in range(point_1[1], point_2[0]+1)]
                    beat_list += [p for p in range(point_2[1], point_3[0]+1)]
                    beat_list += [p for p in range(point_3[1], total_beats+1)]
                    solutions.append((beat_list, reg_sim))

    # only return solutions with highest transition similarity
    if not solutions:
        return None
    else:
        return max(solutions, key=lambda x: x[1])


def less_transitions_algorithm(
    transitions,
    similarities,
    target_beats,
    total_beats,
    beat_analysis
):
    """Simple, greedy algorithm that selects a path with the most transition
    similarity and up to 3 transitions.
    """

    # get beats in measure
    beats_in_measure = int(max([b[1] for b in beat_analysis]))

    # get points as tuples and
    # ensure only points that are in the same position in the measure are used
    points = []
    for current, value in transitions.items():
        for candidate in value:
            if (abs(current-candidate)) % beats_in_measure == 0:
                points.append((current, candidate))

    solutions = Parallel(n_jobs=-2, verbose=0)(delayed(paths_with_up_to_3_transitions)(
        points=points,
        similarities=similarities,
        target_beats=target_beats + offset,
        total_beats=total_beats) for offset in [0, 1, -1, 2, -2, 3, -3])
    if not solutions:
        solutions = Parallel(n_jobs=-2, verbose=0)(delayed(paths_with_up_to_3_transitions)(
            points=points,
            similarities=similarities,
            target_beats=target_beats + offset,
            total_beats=total_beats) for offset in [4, -4, 5, -5, 6, -6, 7, -7])

    solutions = list(filter(None, solutions))

    # find the solution with the highest transition similarity
    try:
        return max(solutions, key=lambda x: x[1])[0]
    except ValueError:
        print("No solutions found, exciting...")
        exit()
