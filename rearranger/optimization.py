"""
Path finding methods for jump points.
"""

import copy


def get_ordinal_encoding(points, beat_no, neighbors=True):
    jumps = {}
    # first possible candidate is neighboring beat
    if neighbors:
        for p in range(beat_no):
            jumps[p] = [p+1]
    for p in points:
        # for now only consider rank 1 points
        if p[1] == 1:
            if p[0][0] in jumps:
                if p[0][1] not in jumps[p[0][0]]:
                    jumps[p[0][0]].append(p[0][1])
            else:
                jumps[p[0][0]] = [p[0][1]]
    return jumps


def get_nonnegative_ordinal_encoding(points, beat_no, neighbors=True):
    jumps = {}
    # first possible candidate is neighboring beat
    if neighbors:
        for p in range(beat_no):
            jumps[p] = [p+1]
    for p in points:
        # only forward jumps, any rank
        if p[0][1] > p[0][0]:
            if p[0][0] in jumps:
                if p[0][1] not in jumps[p[0][0]]:
                    jumps[p[0][0]].append(p[0][1])
            else:
                jumps[p[0][0]] = [p[0][1]]
    return jumps


def get_path(cur_idx, jumps, rem_beats, beat_no, append="> "):

    beat_list = [cur_idx]

    if cur_idx == beat_no and rem_beats == 0:
        # Solution found!
        # print(append + "SOLUTION FOUND!", )
        return beat_list
    elif cur_idx < beat_no and rem_beats > 0:
        targets = copy.deepcopy(jumps[cur_idx])
        # print("Investigate: ", targets)
        for target_idx in targets:
            # print(append + "CUR" + str(cur_idx) + '->' + str(target_idx))
            jumps[cur_idx].remove(target_idx)
            if target_idx != beat_no and cur_idx in jumps[target_idx]:
                jumps[target_idx].remove(cur_idx)

            result = get_path(
                cur_idx=target_idx,
                jumps=copy.deepcopy(jumps),
                rem_beats=rem_beats-1,
                beat_no=beat_no,
                append=append+"> "
            )
            if result:
                # print(beat_list+result)
                return beat_list+result
        # If none returned, no path here
        # print(append + "No targets return solution.")
        return 0
    else:
        # Probably shouldn't be here?
        # print(append + "Hmmm ineligible case?")
        return 0
