import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy_indexed as npi

from schau_mir_in_die_augen.evaluation.top_k import top_k_accuracy_score


class BaseEvaluation():

    def split_fixation_saccades(self, xys, labels, ds):
        """ Generate a list of feature vectors and labels for multiple trajectories

        :param xys: list of trajectories
        :param labels: list of labels
        :return: list of feature matrices for each classifier and list of label matrices
        """

        feats = []
        ls = []
        for xy, y in zip(xys, labels):
            # get a list of features for each classifier. each list contains multiple feature vectors
            Xs, ys = self.trajectory_split_prepare(xy, y, ds)
            # fist iteration, create
            if len(feats) == 0:
                # repeat empty lists
                feats = [pd.DataFrame()] * len(Xs)
                ls = [np.array([])] * len(ys)

            # merge list of features into
            for i, (X, y) in enumerate(zip(Xs, ys)):
                if len(feats[i]) == 0:
                    feats[i] = X
                    ls[i] = y
                else:
                    feats[i] = feats[i].append(X)
                    ls[i] = np.hstack((ls[i], np.asarray(y)))
            
        return feats, ls

    def load_trajectories(self, dataset, limit=None):
        """ Load training and test data

        :param dataset: Dataset instance
        :param partitions: int, default=1, optional
          Split individual trajectories into multiple
        :param limit: int, default=None, optional
          Limit the number of samples (applied before the split)
        :return: list, list, list, list
          train and test trajectories, train and test labels
        """

        X_train, y_train = dataset.load_training()
        X_test, y_test = dataset.load_testing()

        if limit:
            step = len(X_train) // limit
            return X_train[::step], X_test[::step], y_train[::step], y_test[::step]

        return X_train, X_test, y_train, y_test

    def predict_trajetory(self, xy, clfs, ds):
        """ Predict a single trajectory. Uses majority vote on all predictions from all saccades.

        :param xy: ndarray
          2D array of gaze points (x,y)
        :param clfs: classifiers
          list of trained sklean classifier
        :return: list of lists
          list of probability vectors for each sample
        """

        # the class is not needed/available for prediction
        y = 0

        # classify every saccade and fixation
        Xs, _ = self.trajectory_split_prepare(xy, y, ds)
        predicted_probs = []
        for clf, X in zip(clfs, Xs):
            clf_proba = np.mean(clf.predict_proba(X), axis=0) if len(X) > 0 else np.zeros(clf.classes_.shape)
            predicted_probs.append(clf_proba)

        return predicted_probs
    
    def weighted_evaluation(self, X_test, y_test, clfs, clf_names, weights, dataset):
        assert len(clfs) == len(weights)
        assert len(clf_names) == len(clfs)

        y_test = np.asarray(y_test)

        # we need the same classes in all classifiers
        classes_ = np.asarray(clfs[0].classes_)
        for c in clfs:
            assert (classes_ == np.asarray(c.classes_)).all()
        
        print("Evaluating {} test cases".format(len(X_test)))
        start_time = datetime.datetime.now()

        # for each test sample (trajectory), y_hat contains k ndarrays with shape=c elements each.
        # k: number of classifiers for this method
        # c: number of classes
        y_hat = np.asarray([self.predict_trajetory(xy, clfs, dataset) for xy in X_test])

        # weight by sample count per classifier
        per_classifier_count = np.asarray([len(y_hat[:,i]) for i in range(len(weights))])
        total_count = per_classifier_count.sum()
        per_classifier_weight = per_classifier_count / total_count

        # sum of weighted perdictions of all classifiers
        comb_pred = np.asarray([])
        for i, w in enumerate(weights):
            # weighted prediction of the i-th classifier
            weighted = w * y_hat[:,i] * per_classifier_weight[i]
            if len(comb_pred) == 0:
                comb_pred = weighted
            else:
                comb_pred += weighted
        assert comb_pred.shape == (len(y_test), len(clfs[0].classes_))
        #print("Expecting: {}".format(y_test))

        y_comb = classes_[comb_pred.argmax(axis=1)]
        results = {}
        results['Accuracy_unmerged'] = accuracy_score(y_test, y_comb)

        # if there are more test samples per user, combine the predictions
        y_test, comb_pred = npi.group_by(y_test).mean(comb_pred)

        # predictions from all classifiers
        y_comb = classes_[comb_pred.argmax(axis=1)]
        results['Accuracy_combined'] = accuracy_score(y_test, y_comb)

        # for i, w in enumerate(weights):
        #   pred = np.asarray([y[i] for y in y_hat])
        #   y_pred = clfs[i].classes_[pred.argmax(axis=1)]
        #   #print("{} prediprediction: {}".format(clf_names[i], clfs[i].classes_[y_pred]))
        #   print("%s accuracy: %1.3f" % (clf_names[i], accuracy_score(y_test, y_pred)))

        y_true_idxs = np.asarray([np.where(classes_==y_test[i])[0][0] for i in range(len(y_test))])
        top_ks = [top_k_accuracy_score(y_true_idxs, comb_pred, k=i) for i in range(1, len(classes_)+1)]
        results['top-k accuricies'] = top_ks
        print(results)
        print("evaluation time: ", str(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))

        return results
