import pandas as pd

from schau_mir_in_die_augen.datasets.Bioeye import BioEye
from schau_mir_in_die_augen.datasets.rigas import RigasDataset
from schau_mir_in_die_augen.datasets.randomdot import RandomdotDataset
from schau_mir_in_die_augen.datasets.where_humans_look import WHlDataset
from schau_mir_in_die_augen.feature_extraction import all_features
from schau_mir_in_die_augen.rbfn.Rbfn import Rbfn
import schau_mir_in_die_augen.datasets.dataset_helpers as helpers
from schau_mir_in_die_augen.evaluation.base_evaluation import BaseEvaluation

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import datetime
from joblib import cpu_count

from schau_mir_in_die_augen.trajectory_split import sliding_window


class EvaluationWindowed(BaseEvaluation):
    def __init__(self, base_clf, window_size=100):
        self.window_size = window_size
        self.clf = base_clf

    def trajectory_split_prepare(self, xy, label, ds):
        """ Generate feature vectors for all saccades and fixations and our saccade, fixation, general features in a trajectory
        :param xy: ndarray
            2D array of gaze points (x,y)
        :param label: any
             single class this trajectory belongs to
        :param sampleRate: float
            Sensor sample rate in Hz
        :return: list, list
            list of feature vectors (each is a 1D ndarray) and list of classes(these will all be the same)
        """
        angle = helpers.convert_pixel_coordinates_to_angles(xy, *ds.get_screen_params().values())
        smoothed_angle = np.asarray(helpers.savgol_filter_trajectory(angle)).T
        smoothed_pixels = helpers.convert_angles_to_pixel_coordinates(smoothed_angle, *ds.get_screen_params().values())

        window_size = self.window_size
        windows = sliding_window(len(smoothed_pixels), window_size, window_size//2, sample_rate=ds.sample_rate)

        features = pd.DataFrame()
        prev_window = np.asarray([])

        for i, s in enumerate(windows):
            # slice expects start and stop, where stop is exclusive
            part = slice(s[0], s[1] + 1)
            all_feats = all_features(smoothed_pixels[part, :], ds.sample_rate, ds.get_screen_params().values(),
                                     prev_xy=prev_window, omit_stats=False, omit_our=False)
            features = features.append(all_feats)
            prev_window = smoothed_pixels[part, :]

        labels = np.repeat(label, len(features))

        return [features], [labels]

    def train(self, X_train, y_train, ds):
        print("Feature extraction for {} cases".format(len(X_train)))
        start_time = datetime.datetime.now()
        Xs, ys = self.split_fixation_saccades(X_train, y_train, ds)
        print("X_train", Xs[0].shape)
        print("feature extract time: ", str(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))

        print("Training")
        start_time = datetime.datetime.now()
        self.clf.fit(Xs[0], ys[0])
        print("train time: ", str(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))

    def evaluation(self, X_test, y_test, ds):
        return self.weighted_evaluation(X_test, y_test, [self.clf], ['clf'], [1], ds)
