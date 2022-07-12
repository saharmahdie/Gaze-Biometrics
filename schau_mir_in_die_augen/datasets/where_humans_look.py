from itertools import chain

from scipy.io import loadmat
import numpy as np
from os.path import splitext, basename, join as pjoin
import os

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import safe_indexing, indexable

from .BaseDataset import BaseDataset


class WHlDataset(BaseDataset):
    def __init__(self, train_samples, test_samples, random_state=42):
        super().__init__()

        self.random_state = random_state
        self.sample_rate = 240
        self.rec_duration = 3000  # ms
        self.train_samples = train_samples
        self.test_samples = test_samples

        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_folder = '{}/../../data/where_humans_look/DATA/'.format(file_dir)
        self.stim_folder = '{}/../../data/where_humans_look/ALLSTIMULI/'.format(file_dir)

        users = self.get_users()
        cases = self._get_cases()
        dfs = []
        ids = []
        for u in users:
            for c in cases[:len(cases)]:
                df = self.load_data_no_nan(u, c)
                dfs.append(df)
                ids.append(u)

        n_train = train_samples * len(users)
        n_test = test_samples * len(users)
        cv = StratifiedShuffleSplit(test_size=n_test,
                     train_size=n_train,
                     random_state=self.random_state)
        train, test = next(cv.split(X=dfs, y=ids))

        arrays = indexable(dfs, ids)
        self.dfs_train, self.dfs_test, self.ids_train, self.ids_test = list(
            chain.from_iterable((safe_indexing(a, train),
            safe_indexing(a, test)) for a in arrays))

    def get_screen_params(self):
        screen_size_mm = np.asarray([3768, 3015])
        screen_res = np.asarray([1280, 1024])
        return {'pix_per_mm': screen_res / screen_size_mm,
                'screen_dist_mm': 609.6, # appr. 2 feet
                'screen_res': screen_res}

    def get_users(self):
        # name of all folders in DATA dir
        return [basename(f.path)
                for f in os.scandir(self.data_folder)
                if f.is_dir()]

    def _get_cases(self):
        # name of all images without file extension in ALLSTIMULI dir
        return [splitext(basename(f.path))[0]
                for f in os.scandir(self.stim_folder)
                if f.is_file() and splitext(f.path)[1] == '.jpeg']

    def get_stimulus(self, case='i2312141889'):
        return os.path.join(self.stim_folder, '{}.jpeg'.format(case))

    def _load_data(self, user='CNG', case='i2312141889'):
        """ Get x, y array with ordinates for recording $case from $user

        Return:
        2D np.arrays with same lenght and x, y as components
        """
        matdata = loadmat(pjoin(self.data_folder, user, case))
        # matlab structure: e.a. i2312141889.DATA.eyeData
        matlab_case_name = case[:63]  # matlab seems to cap names to 63 letters
        df = matdata[matlab_case_name]['DATA'][0][0]['eyeData'][0, 0]
        # some recordings have 3 columns? just ignore the third
        df = df[:, :2]

        return df

    def load_training(self):
        """ Loads a list of samples
        
        :return: dfs: [(np.array(x,y)], id: [int]
        """

        return self.dfs_train, self.ids_train

    def load_testing(self):
        """ Loads a list of samples
        
        :return: dfs: [(np.array(x,y)], id: [int]
        """
        return self.dfs_test, self.ids_test