"""Path finding methods for transition positions.
"""

import copy
import heapq
import itertools


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


def _get_beats_in_measure(beat_analysis):
    if beat_analysis is None:
        return 1
    return int(max([b[1] for b in beat_analysis]))


def _transition_cost(current, candidate, similarities):
    """Cost model from the paper, using similarity as a smoothness proxy."""
    similarity = similarities.get((current, candidate), 0)
    similarity = max(0, min(1, float(similarity)))

    if candidate > current:
        return 1 - similarity
    return 1 - (similarity / 4)


def _build_adjacency(transitions, similarities, total_beats):
    """Build sparse graph edges.

    Consecutive beat edges consume one output beat and have no transition cost.
    Jump edges consume no output time; their cost reflects cut quality.
    """
    adjacency = {}
    for current in range(total_beats + 1):
        candidates = set(transitions.get(current, []))
        if current < total_beats:
            candidates.add(current + 1)

        edges = []
        for candidate in candidates:
            if candidate == current or candidate < 0 or candidate > total_beats:
                continue

            if candidate == current + 1:
                edges.append((candidate, 1, 0, 0))
            else:
                edges.append((
                    candidate,
                    0,
                    _transition_cost(current, candidate, similarities),
                    1))

        edges.sort(key=lambda edge: (edge[2], edge[3], edge[0]))
        adjacency[current] = edges

    return adjacency


def _reconstruct_path(parent, state):
    path = [state[0]]
    while parent[state] is not None:
        state = parent[state]
        path.append(state[0])
    path.reverse()
    return path


def _path_duration(beats):
    duration = 0
    for current, candidate in zip(beats[:-1], beats[1:]):
        if candidate == current + 1:
            duration += 1
    return duration


def _path_seconds(beats, beat_times):
    duration = 0
    for current, candidate in zip(beats[:-1], beats[1:]):
        if candidate == current + 1:
            duration += beat_times[candidate] - beat_times[current]
    return duration


def sparse_dijkstra_search(
    transitions,
    similarities,
    target_beats,
    total_beats,
    beat_analysis=None,
    tolerance_beats=None,
    max_tolerance_beats=None
):
    """Find a rearrangement path with sparse Dijkstra over beat/time states.

    The search follows the optimization described in the paper: nodes are
    ``(current beat, output beats used)`` states, normal playback edges advance
    time, and transition edges jump to another beat without adding duration.
    """

    beats_in_measure = _get_beats_in_measure(beat_analysis)
    if tolerance_beats is None:
        tolerance_beats = 0
    if max_tolerance_beats is None:
        max_tolerance_beats = max(tolerance_beats, beats_in_measure * 8)

    tolerance_beats = int(max(0, tolerance_beats))
    max_tolerance_beats = int(max(tolerance_beats, max_tolerance_beats))

    min_goal = max(0, target_beats - tolerance_beats)
    max_goal = target_beats + tolerance_beats
    max_output_beats = max(total_beats, target_beats + max_tolerance_beats)

    adjacency = _build_adjacency(transitions, similarities, total_beats)

    start = (0, 0)
    distances = {start: (0, 0)}
    parent = {start: None}
    counter = itertools.count()
    queue = [(0, 0, next(counter), start)]

    best_goal = None
    best_goal_key = None
    fallback_goal = None
    fallback_key = None

    while queue:
        cost, n_jumps, _, state = heapq.heappop(queue)
        if distances.get(state) != (cost, n_jumps):
            continue

        current, duration = state

        if current == total_beats:
            duration_error = abs(duration - target_beats)
            fallback_candidate_key = (duration_error, cost, n_jumps)
            if fallback_key is None or fallback_candidate_key < fallback_key:
                fallback_key = fallback_candidate_key
                fallback_goal = state

            if min_goal <= duration <= max_goal:
                goal_candidate_key = (cost, duration_error, n_jumps)
                if best_goal_key is None or goal_candidate_key < best_goal_key:
                    best_goal_key = goal_candidate_key
                    best_goal = state

            # The rearrangement must end at the final beat.
            continue

        if best_goal_key is not None and cost > best_goal_key[0]:
            break

        for candidate, duration_step, edge_cost, jump_step in adjacency[current]:
            next_duration = duration + duration_step
            if next_duration > max_output_beats:
                continue

            next_state = (candidate, next_duration)
            next_distance = (cost + edge_cost, n_jumps + jump_step)
            if next_distance < distances.get(next_state, (float("inf"), float("inf"))):
                distances[next_state] = next_distance
                parent[next_state] = state
                heapq.heappush(queue, (
                    next_distance[0],
                    next_distance[1],
                    next(counter),
                    next_state))

    final_state = best_goal
    if final_state is None:
        final_state = fallback_goal

    if final_state is None:
        raise ValueError("No path from the first beat to the final beat was found.")

    beat_list = _reconstruct_path(parent, final_state)
    duration = _path_duration(beat_list)
    if duration != final_state[1]:
        raise RuntimeError("Internal path duration accounting failed.")

    return beat_list


def sparse_dijkstra_seconds_search(
    transitions,
    similarities,
    target_seconds,
    total_beats,
    beat_times,
    beat_analysis=None,
    seconds_quantization=0.1,
    tolerance_seconds=0.5,
    max_tolerance_seconds=None,
):
    """Find a rearrangement path by optimizing output duration in seconds.

    This uses the same sparse transition graph as ``sparse_dijkstra_search``,
    but normal playback edges add their actual source beat duration rather
    than one abstract beat. Durations are quantized to keep the state space
    finite while preserving enough precision for second-level targets.
    """

    if seconds_quantization <= 0:
        raise ValueError("seconds_quantization must be positive.")
    beat_times = list(beat_times)
    if len(beat_times) < total_beats + 1:
        raise ValueError("beat_times must include every source beat boundary.")
    beat_time_bins = [
        int(round(float(beat_time) / seconds_quantization))
        for beat_time in beat_times
    ]

    if max_tolerance_seconds is None:
        beats_in_measure = _get_beats_in_measure(beat_analysis)
        mean_beat_duration = beat_times[-1] / max(1, total_beats)
        max_tolerance_seconds = max(tolerance_seconds, beats_in_measure * 8 * mean_beat_duration)

    target_bin = int(round(target_seconds / seconds_quantization))
    tolerance_bin = int(round(tolerance_seconds / seconds_quantization))
    max_output_seconds = max(
        float(beat_times[-1]),
        float(target_seconds) + float(max_tolerance_seconds))
    max_output_bin = int(round(max_output_seconds / seconds_quantization))

    adjacency = _build_adjacency(transitions, similarities, total_beats)

    start = (0, 0)
    distances = {start: (0, 0)}
    parent = {start: None}
    counter = itertools.count()
    queue = [(0, 0, next(counter), start)]

    best_goal = None
    best_goal_key = None

    while queue:
        cost, n_jumps, _, state = heapq.heappop(queue)
        if distances.get(state) != (cost, n_jumps):
            continue

        current, duration_bin = state

        if current == total_beats:
            duration_error = abs(duration_bin - target_bin)
            outside_tolerance = max(0, duration_error - tolerance_bin)
            goal_candidate_key = (outside_tolerance, duration_error, cost, n_jumps)
            if best_goal_key is None or goal_candidate_key < best_goal_key:
                best_goal_key = goal_candidate_key
                best_goal = state
            continue

        for candidate, _, edge_cost, jump_step in adjacency[current]:
            if candidate == current + 1:
                duration_step = max(1, beat_time_bins[candidate] - beat_time_bins[current])
            else:
                duration_step = 0

            next_duration_bin = duration_bin + duration_step
            if next_duration_bin > max_output_bin:
                continue

            next_state = (candidate, next_duration_bin)
            next_distance = (cost + edge_cost, n_jumps + jump_step)
            if next_distance < distances.get(next_state, (float("inf"), float("inf"))):
                distances[next_state] = next_distance
                parent[next_state] = state
                heapq.heappush(queue, (
                    next_distance[0],
                    next_distance[1],
                    next(counter),
                    next_state))

    if best_goal is None:
        raise ValueError("No path from the first beat to the final beat was found.")

    beat_list = _reconstruct_path(parent, best_goal)
    duration_seconds = _path_seconds(beat_list, beat_times)
    if duration_seconds < 0:
        raise RuntimeError("Internal path seconds accounting failed.")

    return beat_list


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
            if point_2[0] < point_1[1]:
                continue
            reg_sim = (similarities[point_1[0], point_1[1]] +
                       similarities[point_2[0], point_2[1]]) / 2.
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
                reg_sim = (similarities[point_1[0], point_1[1]] +
                           similarities[point_2[0], point_2[1]] +
                           similarities[point_3[0], point_3[1]]) / 3.
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
    """Compatibility wrapper for the sparse Dijkstra search.

    Older configs select ``less_transitions``. Keep accepting that value, but
    use the full sparse graph search instead of the bounded three-jump
    heuristic.
    """
    return sparse_dijkstra_search(
        transitions=transitions,
        similarities=similarities,
        target_beats=target_beats,
        total_beats=total_beats,
        beat_analysis=beat_analysis)
