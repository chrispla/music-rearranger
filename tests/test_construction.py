import unittest

import numpy as np

from rearranger.construction import construct_audio


class ConstructAudioTest(unittest.TestCase):
    def test_crossfade_handles_short_intervals(self):
        y = np.linspace(-1, 1, 1000)
        beat_times = np.asarray([0, 0.1, 0.12, 0.2, 0.22, 0.4])
        recon_beats = [0, 1, 4, 5]

        y_new = construct_audio(
            y=y,
            sr=1000,
            recon_beats=recon_beats,
            beat_times=beat_times,
            crossfade=0.1)

        self.assertGreater(len(y_new), 0)
        self.assertFalse(np.any(np.isnan(y_new)))


if __name__ == "__main__":
    unittest.main()
