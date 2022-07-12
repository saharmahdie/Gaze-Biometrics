import numpy as np
from scipy.io import loadmat
from os.path import basename, join as pjoin
import os
import pandas as pd

from .dataset_helpers import convert_angles_to_pixel_coordinates
from .BaseDataset import BaseDataset


class RandomdotDataset(BaseDataset):
    def __init__(self):
        super().__init__()

        self.sample_rate = 1000
        self.rec_duration = 100000 #ms  1 min 40 s
        self.data_folder = '/home/sahar/PycharmProjects/code3/data/randomdot/ds_1_27sub_hor_sacc/'  # this the horz
        self.ver_data_folder = '/home/sahar/PycharmProjects/code3/data/randomdot/ds_1_27sub_ver_sacc/'

        #self.stim_folder = '/home/sahar/PycharmProjects/code3/data/RigasEM/Visual_Stimuli/'

        # Screen dimensions (w × h): 474 × 297 mm
        self.screen_size_mm = np.asarray([474, 297])
        # Screen resolution (w × h): 1680 × 1050 pixels
        self.screen_res = np.asarray([1680, 1050])
        self.pix_per_mm = self.screen_res / self.screen_size_mm
        # Subject's distance from screen: 550 mm
        self.screen_dist_mm = 550
        self.dsname = 'randomdot'

    def get_screen_params(self):
        return {'pix_per_mm': self.pix_per_mm,
                'screen_dist_mm': self.screen_dist_mm,
                'screen_res': self.screen_res}

    def get_users(self):
        # unique users
        return list(sorted(set([basename(f.path)[:5]
                    for f in os.scandir(self.data_folder)
                    if f.is_file()])))

    def _get_cases(self):
        return ['h1', 'h2']

    def get_stimulus(self, case='h1'):
        return os.path.join(self.stim_folder, 'TEXT_Stimulus_Session_{}.png'.format(case[-1]))

    def _load_data(self, user='s_033', case='h1'):
        """ Get x-org, y-org array with ordinates for recording $case from $user

        Return:
        2D np.arrays with equal length x, y as components
        """
        filename = '{}_{}.txt'.format(user, case)
        x = np.genfromtxt(pjoin(self.data_folder, filename), delimiter='    ', skip_header=1 ,usecols=(1))
        y = np.genfromtxt(pjoin(self.data_folder, filename), delimiter='    ', skip_header=1 ,usecols=(2))

        df = np.asarray([x, y]).T

        # convert angles to pixel
        df = convert_angles_to_pixel_coordinates(df, self.pix_per_mm, self.screen_dist_mm, self.screen_res)
        return df


    def load_training(self, partitions=1):
        """ Loads a list of samples

        :param limit: int
        Maximum number of samples per user
        :param partitions: int
        Split each trajectory into $partitions samples

        :return: sampleRate: float, dfs: [(np.array(x,y)], id: [int]
        """
        users = self.get_users()
        cases = self._get_cases()
        dfs = []
        ids = []
        for i, u in enumerate(users):
            c = cases[0]
            df = self.load_data_no_nan(u, c)

            # split samples into $splits parts
            dfs.extend(np.array_split(df, partitions))
            ids.extend(np.repeat(i, partitions).tolist())

        return dfs, ids

    def load_testing(self, partitions=1):
        """ Loads a list of samples

        :param limit: int
        Maximum number of samples per user
        :param partitions: int
        Split each trajectory into $partitions samples

        :return: sampleRate: float, dfs: [(np.array(x,y)], id: [int]
        """
        users = self.get_users()
        cases = self._get_cases()
        dfs = []
        ids = []
        for i, u in enumerate(users):
            c = cases[1]
            df = self.load_data_no_nan(u, c)

            # split samples into $splits parts
            dfs.extend(np.array_split(df, partitions))
            ids.extend(np.repeat(i, partitions).tolist())

        return dfs, ids