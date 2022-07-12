from unittest import TestCase
import numpy as np

from schau_mir_in_die_augen.trajectory_split import sliding_window


class TestSlidingWindow(TestCase):
    def test_sliding_window(self):
        in_len = 5
        w = sliding_window(in_len, 2, 1, sample_rate=1)
        self.assertTrue(np.allclose(list(w), [(0, 1), (1, 2), (2, 3), (3, 4)]))
        w = sliding_window(in_len, 2, 2, sample_rate=1)
        self.assertTrue(np.allclose(list(w), [(0, 1), (2, 3)]))
        w = sliding_window(in_len, 3, 1, sample_rate=1)
        self.assertTrue(np.allclose(list(w), [(0, 2), (1, 3), (2, 4)]))
        w = sliding_window(in_len, 3, 2, sample_rate=1)
        self.assertTrue(np.allclose(list(w), [(0, 2), (2, 4)]))
