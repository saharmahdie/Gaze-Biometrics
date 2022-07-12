import datetime

import numpy as np
import sklearn

from schau_mir_in_die_augen.evaluation.base_evaluation import BaseEvaluation
from schau_mir_in_die_augen.feature_extraction import trajectory_split_prepare_ivt_cached, paper_all_subset


class OurEvaluationAppended(BaseEvaluation):
    def __init__(self, base_clf, vel_threshold=50, min_fix_duration=.1, paper_only=False):
        self.paper_only = paper_only
        self.min_fix_duration = min_fix_duration
        self.vel_threshold = vel_threshold
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
        features_sacc = features[features['is_saccade'] & features['duration'] > 0.012]
        features_fix = features[~features['is_saccade']]
        features_sacc = features_sacc.drop(['is_saccade'], axis=1)
        features_fix = features_fix.drop(['is_saccade'], axis=1)

        if self.paper_only:
            features_fix = paper_all_subset(features_fix)
            features_sacc = paper_all_subset(features_sacc)

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
        return self.weighted_evaluation(X_test, y_test, [self.clf_sac, self.clf_fix], ['sac', 'fix'], [0.5, 0.5], ds)

