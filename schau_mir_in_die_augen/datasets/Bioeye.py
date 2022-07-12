from abc import abstractmethod
from enum import Enum, auto

import numpy as np
from os.path import basename, join as pjoin
import os
import random
from schau_mir_in_die_augen.datasets.dataset_helpers import convert_angles_to_pixel_coordinates
from schau_mir_in_die_augen.datasets.BaseDataset import BaseDataset


class BioEye(BaseDataset):
    class Subsets(Enum):
        RAN_30min_dv = auto()
        TEX_30min_dv = auto()
        RAN_1year_dv = auto()
        TEX_1year_dv = auto()

    def __init__(self, subset=Subsets.TEX_30min_dv, score_level_eval=False, one_year_train=False, user_limit=None, seed=42):
        """

        :param subset: string
        one of: ran30, ran1y, tex30, tex1y
        :param score_level_eval: bool
        see load_testing: evaluation for 1year like in score level paper
        """
        super().__init__()

        self.user_limit = user_limit
        self.seed = seed
        self.one_year_train = one_year_train
        self.score_level_eval = score_level_eval
        self.subset = subset
        self.sample_rate = 250
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_folder = '{}/../../data/BioEye2015_DevSets/{}/'.format(file_dir, subset.name)
        self.stim_folder = '{}/../../data/RigasEM/Visual_Stimuli/'.format(file_dir)

        self.testing_id = '3' if '1year' in subset.name else '1'
        self.training_id = '1' if '1year' in subset.name else '2'
        if one_year_train:
            self.training_id = '3'
            self.testing_id = '1'

        # Screen dimensions (w × h): 474 × 297 mm
        self.screen_size_mm = np.asarray([474, 297])
        # Screen resolution (w × h): 1680 × 1050 pixels
        self.screen_res = np.asarray([1680, 1050])
        self.pix_per_mm = self.screen_res / self.screen_size_mm
        # Subject's distance from screen: 550 mm
        self.screen_dist_mm = 550

    def get_screen_params(self):
        return {'pix_per_mm': self.pix_per_mm,
                'screen_dist_mm': self.screen_dist_mm,
                'screen_res': self.screen_res}

    def get_users(self):
        # unique users
        users = list(sorted(set([basename(f.path)[:6]
                         for f in os.scandir(self.data_folder)
                         if f.is_file()])))

        if not (self.user_limit is None):
            random.seed(self.seed)
            random.shuffle(users)
            users = users[:self.user_limit]

        return users

    def _get_cases(self):
        return [self.training_id, self.testing_id]

    def get_stimulus(self, case="1"):
        if self.subset in {BioEye.Subsets.RAN_30min_dv, BioEye.Subsets.RAN_1year_dv}:
            return os.path.join(self.stim_folder, 'RAN_Stimulus.png')
        else:
            return os.path.join(self.stim_folder, 'TEXT_Stimulus_Session_{}.png'.format(case))

    def _load_data(self, user='ID_001', case='1'):
        """ Get x-org, y-org array with ordinates for recording $case from $user

        Return:
        2D np.arrays with equal length x, y as components
        """
        filename = '{}_{}.txt'.format(user, case)
        xy = np.genfromtxt(pjoin(self.data_folder, filename), skip_header=1, usecols=(1,2))

        # convert angles to pixel
        df = convert_angles_to_pixel_coordinates(xy, self.pix_per_mm, self.screen_dist_mm, self.screen_res)

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
        dfs = []
        ids = []
        for u in users:
            df = self.load_data_no_nan(u, case=self.training_id)

            # split samples into $splits parts
            dfs.extend(np.array_split(df, partitions))
            ids.extend(np.repeat(u, partitions).tolist())

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
        dfs = []
        ids = []

        # it seems like the a score level authors used both sessions for evaluation - even though the first session is
        # the same as the first from the first recording session
        if self.score_level_eval:
            assert self.testing_id != '1'
            for u in users:
                df = self.load_data_no_nan(u, case="1")

                # split samples into $splits parts
                dfs.extend(np.array_split(df, partitions))
                ids.extend(np.repeat(u, partitions).tolist())

        for u in users:
            df = self.load_data_no_nan(u, case=self.testing_id)
            # split samples into $splits parts
            dfs.extend(np.array_split(df, partitions))
            ids.extend(np.repeat(u, partitions).tolist())

        return dfs, ids

