import datetime
import pickle

from joblib import cpu_count
from sklearn.ensemble import RandomForestClassifier

from schau_mir_in_die_augen.evaluation.evaluation_our_appended import OurEvaluationAppended
from schau_mir_in_die_augen.evaluation.evaluation_our_one_rf import OurEvaluationOne
from schau_mir_in_die_augen.evaluation.evaluation_score_level import ScoreLevelEvaluation
from schau_mir_in_die_augen.evaluation.evaluation_windowed import EvaluationWindowed
from schau_mir_in_die_augen.rbfn.Rbfn import Rbfn
from scripts.arguments import get_dataset, parse_config


def main(args):
    # dataset selection
    ds = get_dataset(args.dataset, args)

    # initialize the right classifier
    clf = None
    if args.clf == 'rf':
        clf = RandomForestClassifier(n_estimators=args.rf_n_estimators, random_state=args.seed, n_jobs=cpu_count())
    elif args.clf == 'rbfn':
        clf = Rbfn(args.rbfn_k, random_state=args.seed)
    else:
        print(f'ERROR: unknown classifier: {args.clf}')

    # init the right method
    eva = None
    if args.method == 'score-level':
        text_features = 'tex' in args.dataset
        eva = ScoreLevelEvaluation(clf, text_features=text_features, vel_threshold=args.ivt_threshold, min_fix_duration=args.ivt_min_fix_time)
    elif args.method == 'our-append':
        eva = OurEvaluationAppended(clf, vel_threshold=args.ivt_threshold, min_fix_duration=args.ivt_min_fix_time)
    elif args.method == 'paper-append':
        eva = OurEvaluationAppended(clf, vel_threshold=args.ivt_threshold, min_fix_duration=args.ivt_min_fix_time, paper_only=True)
    elif args.method == 'our-one-clf':
        eva = OurEvaluationOne(clf, vel_threshold=args.ivt_threshold, min_fix_duration=args.ivt_min_fix_time)
    elif args.method == 'our-windowed':
        eva = EvaluationWindowed(clf, args.window_size)
        print("WARNING: windowed evaluation currently ignores the dataset")
    else:
        print(f'ERROR: unknown method{args.method}')

    # training
    start_time = datetime.datetime.now()
    X_train, X_test, y_train, y_test = eva.load_trajectories(ds)
    eva.train(X_train, y_train, ds)

    with open(args.modelfile, 'wb') as f:
        pickle.dump(eva, f)

    print("total time: ", str(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))


if __name__ == '__main__':
    pargs = parse_config()
    print(pargs)
    main(pargs)
