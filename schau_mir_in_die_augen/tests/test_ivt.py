from unittest import TestCase
import numpy as np

import schau_mir_in_die_augen.features as feat
import schau_mir_in_die_augen.trajectory_split


class TestIvt(TestCase):
    def setUp(self):
        #    0  1  2  3    4    5
        x = [0, 0, 0, 0, 0.1, 0.2]
        y = [0, 0, 0, 0, 0.1, 0.2]
        xy =  np.asarray([x, y]).T
        self.velocities = feat.calculate_distance_vector(xy)

    def test_ivt_1_1(self):
        # fixation and sacade
        s, f = schau_mir_in_die_augen.trajectory_split.ivt(self.velocities, .01, min_fix_duration=1, sampleRate=1)
        self.assertTrue(np.allclose(list(s), [(3, 4)]))
        self.assertTrue(np.allclose(list(f), [(0, 2)]))

    def test_ivt_dur_equal(self):
        # min duration exactly equal to duration
        s, f = schau_mir_in_die_augen.trajectory_split.ivt(self.velocities, .01, min_fix_duration=3, sampleRate=1)
        self.assertTrue(np.allclose(list(s), [(3, 4)]))
        self.assertTrue(np.allclose(list(f), [(0, 2)]))

    def test_ivt_dur_larger(self):
        # min duration larger than smallest fixation -> only one saccade
        s, f = schau_mir_in_die_augen.trajectory_split.ivt(self.velocities, .01, min_fix_duration=4, sampleRate=1)
        self.assertTrue(np.allclose(list(s), [(0, 4)]))
        self.assertEqual(len(list(f)), 0)

        # short fixation at the end
        s, f = schau_mir_in_die_augen.trajectory_split.ivt(np.flip(self.velocities, 0), .01, min_fix_duration=4, sampleRate=1)
        self.assertTrue(np.allclose(list(s), [(0, 4)]))
        self.assertEqual(len(list(f)), 0)

    def test_ivt_single(self):
        # only saccade
        #    0  1  2  3  4  5
        x = [1, 2, 3, 4, 5, 6]
        y = [1, 2, 3, 4, 5, 6]
        xy =  np.asarray([x, y]).T
        velocities = feat.calculate_distance_vector(xy)
        s, f = schau_mir_in_die_augen.trajectory_split.ivt(velocities, .01, min_fix_duration=1, sampleRate=1)
        self.assertTrue(np.allclose(list(s), [(0, 4)]))
        self.assertEqual(len(list(f)), 0)

        # only fixx
        #    0  1  2  3
        x = [0, 0, 0, 0]
        y = [0, 0, 0, 0]
        xy =  np.asarray([x, y]).T
        velocities = feat.calculate_distance_vector(xy)
        s, f = schau_mir_in_die_augen.trajectory_split.ivt(velocities, .01, min_fix_duration=1, sampleRate=1)
        self.assertEqual(len(list(s)), 0)
        self.assertTrue(np.allclose(list(f), [(0, 2)]))

    def test_ivt_complex(self):
        # saccade in the middle
        #    0  1  2  3  4  5  6  7
        x = [0, 0, 0, 3, 4, 0, 0, 0]
        y = [0, 0, 0, 3, 4, 0, 0, 0]
        xy =  np.asarray([x, y]).T
        velocities = feat.calculate_distance_vector(xy)
        s, f = schau_mir_in_die_augen.trajectory_split.ivt(velocities, .01, min_fix_duration=1, sampleRate=1)
        self.assertTrue(np.allclose(list(s), [(2, 4)]))
        self.assertTrue(np.allclose(list(f), [(0, 1), (5, 6)]))

        # # saccade in the middle, fixations too short
        s, f = schau_mir_in_die_augen.trajectory_split.ivt(velocities, .01, min_fix_duration=3, sampleRate=1)
        self.assertEqual(len(list(f)), 0)
        self.assertTrue(np.allclose(list(s), [(0, 6)]))

    def test_ivt_sacc_start_end(self):
        # test sequences with uneven amount of fixations and saccades
        x = [3, 4, 0, 0, 0, 3, 4]
        y = [3, 4, 0, 0, 0, 3, 4]
        xy =  np.asarray([x, y]).T
        velocities = feat.calculate_distance_vector(xy)
        s, f = schau_mir_in_die_augen.trajectory_split.ivt(velocities, .01, min_fix_duration=1, sampleRate=1)
        self.assertTrue(np.allclose(list(s), [(0, 1), (4, 5)]))
        self.assertTrue(np.allclose(list(f), [(2, 3)]))
