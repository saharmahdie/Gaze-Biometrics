import datetime

import numpy as np

from schau_mir_in_die_augen.evaluation.base_evaluation import BaseEvaluation
from schau_mir_in_die_augen.feature_extraction import trajectory_split_prepare_ivt_cached


class OurEvaluationOne(BaseEvaluation):

    def __init__(self, base_clf, vel_threshold=50, min_fix_duration=.1):
        self.min_fix_duration = min_fix_duration
        self.vel_threshold = vel_threshold
        self.clf = base_clf

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
        #features = features.drop(['is_saccade'], axis=1)

        labels_sacc = np.repeat(label, len(features))
        return [features], [labels_sacc]

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
