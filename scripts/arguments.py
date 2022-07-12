import argparse
import os
import sys

from schau_mir_in_die_augen.datasets.Bioeye import BioEye
from schau_mir_in_die_augen.datasets.rigas import RigasDataset
from schau_mir_in_die_augen.datasets.where_humans_look import WHlDataset

datasets = ['bio-tex', 'bio-tex1y', 'bio-ran', 'bio-ran1y', 'whl', 'rigas-tex', 'rigas-ran']


def get_dataset(dataset_name, args):
    """
    Construct dataset with individual configuration
    :param dataset_name: from the datasets array
    :param args: configuration from parse_config()
    :return: dataset object
    """
    ds = None
    if dataset_name == 'bio-tex':
        ds = BioEye(BioEye.Subsets.TEX_30min_dv, user_limit=args.ul)
    elif dataset_name == 'bio-ran':
        ds = BioEye(BioEye.Subsets.RAN_30min_dv, user_limit=args.ul)
    elif dataset_name == 'bio-tex1y':
        ds = BioEye(BioEye.Subsets.TEX_1year_dv, score_level_eval=args.score_level_1y,
                    one_year_train=args.score_level_1y_train, user_limit=args.ul)
    elif dataset_name == 'bio-ran1y':
        ds = BioEye(BioEye.Subsets.RAN_1year_dv, score_level_eval=args.score_level_1y,
                    one_year_train=args.score_level_1y_train, user_limit=args.ul)
    elif dataset_name == 'whl':
        ds = WHlDataset(args.whl_train_samples, args.whl_test_samples, random_state=args.seed)
    elif dataset_name == 'rigas-tex':
        ds = RigasDataset()
    elif dataset_name == 'rigas-ran':
        print('# TODO: implement rigas dataset selection as in bioeye')
    else:
        print(f'ERROR: unknown dataset: {dataset_name}')

    return ds


def parse_config():
    """
    Configuration from command line arguments
    :return: namespace with configuration
    """
    parser = argparse.ArgumentParser(description='Entry point for evaluations')
    parser.add_argument('--method',
                        choices=['score-level', 'our-append', 'paper-append', 'our-one-clf', 'our-windowed'],
                        required=True)
    parser.add_argument('--dataset', choices=datasets, required=True)
    parser.add_argument('--clf', choices=['rf', 'rbfn'], required=True)
    # optional parameters - used depending on the method
    parser.add_argument('--test_dataset', choices=datasets, required=False)
    parser.add_argument('--rf_n_estimators', type=int, default=400)
    parser.add_argument('--rbfn_k', type=int, default=32)
    parser.add_argument('--window_size', type=int, default=100, help='Window size in seconds for windowed evaluation.')
    parser.add_argument('--ivt_threshold', type=int, default=50, help='Velocity threshold for ivt.')
    parser.add_argument('--ivt_min_fix_time', type=float, default=0.1,
                        help='Minimum duration to count as fixation for ivt.')
    parser.add_argument('--whl_train_samples', type=int, default=1,
                        help='Training samples used in WHL dataset per user.')
    parser.add_argument('--whl_test_samples', type=int, default=1,
                        help='Testing samples used in WHL dataset per user.')
    parser.add_argument('--score_level_1y', action='store_true')
    parser.add_argument('--score_level_1y_train', action='store_true')
    parser.add_argument('--ul', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--modelfile', required=False)
    args = parser.parse_args()

    # check and infer arguments
    if not args.modelfile:
        args.modelfile = f'../models/{args.method}_{args.dataset}_{args.clf}.pickle'
    else:
        args.modelfile = args.modelfile

    # early check if target directories exist
    if not os.path.isdir(os.path.dirname(args.modelfile)):
        print(f"Model dir does not exist: {os.path.dirname(args.modelfile)}")
        sys.exit(-1)

    return args
