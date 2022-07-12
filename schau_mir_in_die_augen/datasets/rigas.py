import numpy as np
from scipy.io import loadmat
from os.path import basename, join as pjoin
import os

from .dataset_helpers import convert_angles_to_pixel_coordinates
from .BaseDataset import BaseDataset


class RigasDataset(BaseDataset):
    def __init__(self):
        super().__init__()

        self.sample_rate = 1000
        self.rec_duration = 60000 #ms
        # self.data_folder = '/home/sahar/PycharmProjects/code3/data/RigasEM/EM_Data/emfe_mat_files/'
        # self.stim_folder = '/home/sahar/PycharmProjects/code3/data/RigasEM/Visual_Stimuli/'
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_folder = '{}/../../data/RigasEM/EM_Data/emfe_mat_files/'.format(file_dir)
        self.stim_folder = '{}/../../data/RigasEM/Visual_Stimuli/'.format(file_dir)

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
        return list(sorted(set([basename(f.path)[:5]
                    for f in os.scandir(self.data_folder)
                    if f.is_file()])))

    def _get_cases(self):
        return ['S1', 'S2']

    def get_stimulus(self, case='S1'):
        return os.path.join(self.stim_folder, 'TEXT_Stimulus_Session_{}.png'.format(case[-1]))

    def _load_data(self, user='S_001', case='S1'):
        """ Get x-org, y-org array with ordinates for recording $case from $user

        Return:
        2D np.arrays with equal length x, y as components
        """
        filename = '{}_{}_TEX.mat'.format(user, case)
        matdata = loadmat(pjoin(self.data_folder, filename))
        x = matdata['ET']['Vectors'][0][0]['Xorg'][0][0]
        y = -matdata['ET']['Vectors'][0][0]['Yorg'][0][0]
        df = np.asarray([x, y]).T[0]

        # convert angles to pixel
        df = convert_angles_to_pixel_coordinates(df, self.pix_per_mm, self.screen_dist_mm, self.screen_res)

        return df

    def _load_smoothed_angle_data(self, user='S_001', case='S1'):
        """ Get smoothed x-org, y-org array with velocity components for recording $case from $user

        Return:
        2D np.arrays with equal length and Xsmo, Ysmo as components
        """
        filename = '{}_{}_TEX.mat'.format(user, case)
        matdata = loadmat(pjoin(self.data_folder, filename))
        vx = matdata['ET']['Vectors'][0][0]['Xsmo'][0][0]
        vy = -matdata['ET']['Vectors'][0][0]['Ysmo'][0][0]
        df = np.asarray([vx, vy]).T[0]

        return df

    def _load_velocity_data(self, user='S_001', case='S1'):
        """ Get x-vel, y-vel array with velocity components for recording $case from $user

        Return:
        2D np.arrays with equal length and x-vel, y-vel as components
        """
        filename = '{}_{}_TEX.mat'.format(user, case)
        matdata = loadmat(pjoin(self.data_folder, filename))
        vx = matdata['ET']['Vectors'][0][0]['velX'][0][0]
        vy = -matdata['ET']['Vectors'][0][0]['velY'][0][0]
        df = np.asarray([vx, vy]).T[0]

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