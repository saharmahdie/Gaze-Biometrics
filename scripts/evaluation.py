import os
import socket
import datetime
import json
import pickle
import subprocess

from scripts.arguments import parse_config, get_dataset


def main(args):
    # dataset selection
    ds = get_dataset(args.dataset, args)

    if not args.modelfile:
        modelfile = f'../models/{args.method}_{args.dataset}_{args.clf}.pickle'
    else:
        modelfile = args.modelfile
    with open(modelfile, 'rb') as f:
        try:
            eva = pickle.load(f)
        except:
            print(f'Failed loading evaluation: {modelfile}')
            return -1

    # data for training, we load them to check whether all testing labels are known from training
    start_time = datetime.datetime.now()
    X_train, X_test, y_train, y_test = eva.load_trajectories(ds)

    test_ds = ds
    if args.test_dataset:
        print('Evaluating on different dataset!')
        test_ds = get_dataset(args.test_dataset, args)
        assert type(ds) == type(test_ds), 'Completely different datasets have different IDs and cannot be used in transfer eval'

        _, X_test, _, y_test = eva.load_trajectories(test_ds)

        if not set(test_ds.get_users()).issubset(set(ds.get_users())):
            print('Warning: Not all test labels are present in the training labels. Using a common subset.')
        train_labels = set(y_train)
        # if we train with fewer labels than present in the test set, test only on samples that are present in the train set
        X_test, y_test = zip(*[(x, y) for x, y in zip(X_test, y_test) if y in train_labels])
    print("data loading time: ", str(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))

    # evaluation
    # really hacky way to do all the evals. we can make this much smarter!
    sessions_per_subject = len(X_test) // len(test_ds.get_users())
    # 1 to 10 -> step 1
    for s in range(1, 11):
        if s > sessions_per_subject:
            continue
        print(f'Samples per subject: {s}')
        eval_subset(X_test, y_test, test_ds, eva, args, s)
    # 20 to 60 -> step 10
    for s in range(20, 70, 10):
        if s > sessions_per_subject:
            continue
        print(f'Samples per subject: {s}')
        eval_subset(X_test, y_test, test_ds, eva, args, s)
    # 100 to max -> step 100
    for s in range(100, sessions_per_subject+100, 100):
        if s > sessions_per_subject:
            continue
        print(f'Samples per subject: {s}')
        eval_subset(X_test, y_test, test_ds, eva, args, s)

    print("total time: ", str(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))


def eval_subset(X_test, y_test, test_ds, eva, args, subset_lengh):
    subtest_x = []
    subtest_y = []
    classes = sorted(list(set(y_test)))
    for c in classes:
        class_samples = [x for x, y in zip(X_test, y_test) if y == c]
        subtest_x.extend(class_samples[:subset_lengh])
        subtest_y.extend([c] * subset_lengh)
    res = eva.evaluation(subtest_x, subtest_y, test_ds)
    args.whl_test_samples = subset_lengh
    res['config'] = vars(args)
    git_commit = subprocess.check_output(["git", "describe", "--always"]).strip()
    result_name = f'{socket.gethostname()}_{git_commit}_{str(datetime.datetime.now())}'
    filename = f'results/{result_name}.json'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as outfile:
        json.dump(res, outfile)


if __name__ == '__main__':
    pargs = parse_config()
    print(pargs)
    main(pargs)
