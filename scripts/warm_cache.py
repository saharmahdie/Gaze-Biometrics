from joblib import cpu_count, Parallel, delayed
from sklearn.ensemble import RandomForestClassifier

from schau_mir_in_die_augen.datasets.Bioeye import BioEye
from schau_mir_in_die_augen.evaluation.evaluation_score_level import ScoreLevelEvaluation

datasets = [BioEye(BioEye.Subsets.TEX_30min_dv), BioEye(BioEye.Subsets.RAN_30min_dv),
            BioEye(BioEye.Subsets.TEX_1year_dv, score_level_eval=True),
            BioEye(BioEye.Subsets.RAN_1year_dv, score_level_eval=True)]

def warm(ds):
    clf = RandomForestClassifier(n_estimators=1, max_depth=1, max_features=1, random_state=42, n_jobs=cpu_count())
    eva = ScoreLevelEvaluation(clf, text_features=False)
    X_train, X_test, y_train, y_test = eva.load_trajectories(ds)
    eva.train(X_train, y_train, ds)
    eva.evaluation(X_test, y_test, ds)

#Parallel(n_jobs=4)(delayed(warm)(ds) for ds in datasets)
[warm(ds) for ds in datasets]