import unittest
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import unique_labels

from schau_mir_in_die_augen.datasets.Bioeye import BioEye
from schau_mir_in_die_augen.datasets.where_humans_look import WHlDataset
from schau_mir_in_die_augen.evaluation.base_evaluation import BaseEvaluation
from schau_mir_in_die_augen.evaluation.evaluation_score_level import ScoreLevelEvaluation

class DummyClf(BaseEstimator):

    def __init__(self):
        self.counter = 0
        self.classes_ = []

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        return self

    def predict_proba(self, X):
        p = np.asarray(np.arange(len(self.classes_))) / len(self.classes_)
        p = p + self.counter
        self.counter = self.counter + 1
        return np.asarray([p for _ in range(len(X))])

#
#
# class DummyEvaluator(BaseEvaluation):
#     def __init__(self):
#         self.curr_pred = 0
#
#     def predict_trajetory(self, xy, clfs, ds):
#         predicted_probs = [np.array([0.9, 0.1]), np.array([0.2, 0.8])]
#
#         return predicted_probs
#
#     def evaluation(self, X_test, y_test, ds):
#         # Evaluation
#         self.weighted_evaluation(X_test, y_test, [self.clf_sac, self.clf_fix], ['sac', 'fix'], [0.5, 0.5], ds)

class TestDatasets(unittest.TestCase):

    def test_eval(self):
        pass
        # I am leaving this here for further testing
        # #ds = BioEye()
        # ds = WHlDataset()
        # clf = DummyClf()
        # eva = ScoreLevelEvaluation(clf, vel_threshold=150)
        # X_train, X_test, y_train, y_test = eva.load_trajectories(ds)
        # eva.train(X_train, y_train, ds)
        # eva.evaluation(X_test, y_test, ds)