import numpy as np

from .dataset_helpers import nan_helper


class BaseDataset:

    def __init__(self):
        self.sample_rate = 0

    def get_screen_params(self):
        return {'pix_per_mm': np.asarray([0, 0]),
                'screen_dist_mm': 0,
                'screen_res': np.asarray([0, 0])}

    def get_users(self):
        """ Unique list of users

        :return: list of users
        """
        return []

    def _get_cases(self):
        """ List of recordings per user

        :return: list of recordings per user
        """
        return []

    def _load_data(self, user, case):
        """ Get x, y array with ordinates for recording $case from $user

        Return:
        2D np.arrays with same lenght and x, y as components
        """

        return np.asarray([])

    def get_stimulus(self, case='S1'):
        """ Relative filename of stimulus

        :param case: case from get_cases
        :return: string
           Relative path to stimulus
        """
        return ""

    def load_data_no_nan(self, user, case):
        """ Wraps _load_data and replaces NaNs by interpolation
        """
        df = self._load_data(user, case)
        # dataset contains nans, interpolate linearly
        nans_x, x = nan_helper(df[:, 0])
        df[nans_x, 0] = np.interp(x(nans_x), x(~nans_x), df[~nans_x, 0])
        nans_y, y = nan_helper(df[:, 1])
        df[nans_y, 1] = np.interp(y(nans_y), y(~nans_y), df[~nans_y, 1])

        return df
