import unittest
import numpy as np

import schau_mir_in_die_augen.datasets.dataset_helpers as helpers
from schau_mir_in_die_augen.features import calculate_velocity
from schau_mir_in_die_augen.datasets.Bioeye import BioEye


class TestDatasets(unittest.TestCase):

    def setUp(self):
        self.xy = np.random.randint(-100, 1500, (30, 2))

    def test_pixel_angle_convert(self):
        x = [0, 0, 1, 2, 0, 3]
        y = [0, 0, 0, 0, 0, 3]
        xy = np.asarray([x, y]).T
        screen_size_mm = np.asarray([474, 297])
        screen_res = np.asarray([1680, 1050])
        pix_per_mm = screen_res / screen_size_mm
        screen_dist_mm = 550
        params = [pix_per_mm, screen_dist_mm, screen_res]
        a = helpers.convert_angles_to_pixel_coordinates(helpers.convert_pixel_coordinates_to_angles(xy, *params), *params)
        self.assertTrue(np.allclose(xy, a))
        b = helpers.convert_pixel_coordinates_to_angles(helpers.convert_angles_to_pixel_coordinates(xy, *params), *params)
        self.assertTrue(np.allclose(b, xy))

    def test_savgol_filter(self):
        ds = BioEye()
        xy = ds._load_data()
        angle = helpers.convert_pixel_coordinates_to_angles(xy, *ds.get_screen_params().values())

