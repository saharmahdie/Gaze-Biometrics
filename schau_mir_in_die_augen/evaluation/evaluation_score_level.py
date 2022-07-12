import datetime

import numpy as np
import sklearn
from joblib import cpu_count
from sklearn.ensemble import RandomForestClassifier

from schau_mir_in_die_augen.datasets.Bioeye import BioEye
from schau_mir_in_die_augen.evaluation.base_evaluation import BaseEvaluation
from schau_mir_in_die_augen.feature_extraction import sac_subset, fix_subset, \
    trajectory_split_prepare_ivt_cached
from schau_mir_in_die_augen.rbfn.Rbfn import Rbfn


class ScoreLevelEvaluation(BaseEvaluation):
    def __init__(self, base_clf, text_features=False, vel_threshold=50, min_fix_duration=.1):
        self.min_fix_duration = min_fix_duration
        self.vel_threshold = vel_threshold
        self.text_features = text_features
        self.clf_fix = sklearn.base.clone(base_clf)
        self.clf_sac = sklearn.base.clone(base_clf)

    def trajectory_split_prepare(self, xy, label, ds):
        """ Generate feature vectors for all saccades and fixations in a trajectory

        :param xy: ndarray
            2D array of gaze points (x,y)
        :param label: any
             single class this trajectory belongs to
        :return: list, list
            list of feature vectors (each is a 1D ndarray) and list of classes(these will all be the same)
        """
        features = trajectory_split_prepare_ivt_cached(xy, ds, self.vel_threshold, self.min_fix_duration)

        # filter features we use
        features_sacc = features[features['is_saccade'] & features['duration']>0.012]
        features_fix = features[~features['is_saccade']]
        features_sacc = sac_subset(features_sacc, self.text_features)
        features_fix = fix_subset(features_fix, self.text_features)

        labels_sacc = np.repeat(label, len(features_sacc))
        labels_fix = np.repeat(label, len(features_fix))
        return [features_sacc, features_fix], [labels_sacc, labels_fix]

    def train(self, X_train, y_train, ds):
        print("Feature extraction for {} cases".format(len(X_train)))
        start_time = datetime.datetime.now()
        Xs, ys = self.split_fixation_saccades(X_train, y_train, ds)
        print("X_train_sac", Xs[0].shape)
        print("X_train_fix", Xs[1].shape)
        print("feature extract time: ", str(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))

        print("Training")
        start_time = datetime.datetime.now()
        self.clf_sac.fit(Xs[0], ys[0])
        self.clf_fix.fit(Xs[1], ys[1])
        print("train time: ", str(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))

    def evaluation(self, X_test, y_test, ds):
        # Evaluation
        return self.weighted_evaluation(X_test, y_test, [self.clf_sac, self.clf_fix], ['sac', 'fix'], [0.5, 0.5], ds)
