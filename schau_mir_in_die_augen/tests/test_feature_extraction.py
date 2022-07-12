import unittest
import numpy as np

from schau_mir_in_die_augen.feature_extraction import all_features, fix_subset, sac_subset


class TestSaccades(unittest.TestCase):

    def setUp(self):
        self.xy = np.random.randint(-100, 1500, (30, 2))
        self.screen_params = {'pix_per_mm': np.asarray([3.5443037974683542, 3.5443037974683542]),
                              'screen_dist_mm': 550.,
                              'screen_res': np.asarray([1680, 1050])}

    def test_statistics(self):
        all_feats = all_features(self.xy, 1, self.screen_params.values())
        #print(all_feats.columns)
        self.assertEqual(len(fix_subset(all_feats, True).columns), 9)
        self.assertEqual(len(fix_subset(all_feats, False).columns), 9)

        self.assertEqual(len(sac_subset(all_feats, True).columns), 43)
        self.assertEqual(len(sac_subset(all_feats, False).columns), 40)

