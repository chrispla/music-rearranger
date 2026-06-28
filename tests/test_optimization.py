import unittest

from rearranger.optimization import (
    get_transitions,
    less_transitions_algorithm,
    sparse_dijkstra_search,
    sparse_dijkstra_seconds_search,
)


def path_duration(path):
    return sum(1 for current, target in zip(path[:-1], path[1:]) if target == current + 1)


class SparseDijkstraSearchTest(unittest.TestCase):
    def test_forward_jump_shortens_to_target(self):
        transitions, similarities = get_transitions(
            points=[([4, 8], 0.9)],
            n_beats=12,
            type="ideal",
            neighbors=True)

        path = sparse_dijkstra_search(
            transitions=transitions,
            similarities=similarities,
            target_beats=8,
            total_beats=12,
            beat_analysis=[[0, 1], [1, 2], [2, 3], [3, 4]],
            tolerance_beats=0,
            max_tolerance_beats=0)

        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 12)
        self.assertIn((4, 8), list(zip(path[:-1], path[1:])))
        self.assertEqual(path_duration(path), 8)

    def test_backward_jump_extends_to_target(self):
        transitions, similarities = get_transitions(
            points=[([6, 2], 1.0)],
            n_beats=8,
            type="ideal",
            neighbors=True)

        path = sparse_dijkstra_search(
            transitions=transitions,
            similarities=similarities,
            target_beats=12,
            total_beats=8,
            beat_analysis=[[0, 1], [1, 2], [2, 3], [3, 4]],
            tolerance_beats=0,
            max_tolerance_beats=0)

        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 8)
        self.assertIn((6, 2), list(zip(path[:-1], path[1:])))
        self.assertEqual(path_duration(path), 12)

    def test_more_than_three_transitions_are_allowed(self):
        points = [
            ([2, 5], 1.0),
            ([7, 10], 1.0),
            ([12, 15], 1.0),
            ([17, 20], 1.0),
        ]
        transitions, similarities = get_transitions(
            points=points,
            n_beats=20,
            type="ideal",
            neighbors=True)

        path = sparse_dijkstra_search(
            transitions=transitions,
            similarities=similarities,
            target_beats=8,
            total_beats=20,
            beat_analysis=[[0, 1], [1, 2], [2, 3], [3, 4]],
            tolerance_beats=0,
            max_tolerance_beats=0)

        jumps = [edge for edge in zip(path[:-1], path[1:]) if edge[1] != edge[0] + 1]
        self.assertEqual(jumps, [(2, 5), (7, 10), (12, 15), (17, 20)])
        self.assertEqual(path_duration(path), 8)

    def test_falls_back_to_closest_reachable_duration(self):
        transitions, similarities = get_transitions(
            points=[],
            n_beats=10,
            type="ideal",
            neighbors=True)

        path = sparse_dijkstra_search(
            transitions=transitions,
            similarities=similarities,
            target_beats=6,
            total_beats=10,
            beat_analysis=[[0, 1], [1, 2], [2, 3], [3, 4]],
            tolerance_beats=0,
            max_tolerance_beats=0)

        self.assertEqual(path, list(range(11)))
        self.assertEqual(path_duration(path), 10)

    def test_legacy_less_transitions_uses_sparse_search(self):
        transitions, similarities = get_transitions(
            points=[([4, 8], 0.9)],
            n_beats=12,
            type="ideal",
            neighbors=True)

        path = less_transitions_algorithm(
            transitions=transitions,
            similarities=similarities,
            target_beats=8,
            total_beats=12,
            beat_analysis=[[0, 1], [1, 2], [2, 3], [3, 4]])

        self.assertEqual(path_duration(path), 8)


class SparseDijkstraSecondsSearchTest(unittest.TestCase):
    def test_forward_jump_skips_slow_region_to_hit_seconds(self):
        transitions, similarities = get_transitions(
            points=[([1, 2], 1.0)],
            n_beats=3,
            type="ideal",
            neighbors=True)

        path = sparse_dijkstra_seconds_search(
            transitions=transitions,
            similarities=similarities,
            target_seconds=2,
            total_beats=3,
            beat_times=[0, 1, 11, 12],
            beat_analysis=[[0, 1], [1, 2], [11, 3], [12, 4]],
            seconds_quantization=0.1)

        self.assertEqual(path, [0, 1, 2, 3])

    def test_backward_jump_extends_to_target_seconds(self):
        transitions, similarities = get_transitions(
            points=[([2, 1], 1.0)],
            n_beats=3,
            type="ideal",
            neighbors=True)

        path = sparse_dijkstra_seconds_search(
            transitions=transitions,
            similarities=similarities,
            target_seconds=4,
            total_beats=3,
            beat_times=[0, 1, 2, 3],
            beat_analysis=[[0, 1], [1, 2], [2, 3], [3, 4]],
            seconds_quantization=0.1)

        self.assertEqual(path, [0, 1, 2, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
