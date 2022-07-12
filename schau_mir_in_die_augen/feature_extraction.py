""" Combination and application of the Features from features.py """

import numpy as np
import pandas as pd
from joblib import Memory
from scipy.stats import skew, kurtosis

from schau_mir_in_die_augen.datasets import dataset_helpers
from schau_mir_in_die_augen.features import calculate_velocity, statistics, angular_acceleration, \
    calculate_distance_vector, total_length, angle_between_first_and_last_points, \
    distance_between_first_and_last_points, calculate_dispersion, \
    distance_from_previous_fix_or_sacc, stat_names, \
    general_gaze_points_features, acceleration_features, acceleration_features_names, \
    direction_changes_names, \
    direction_changes, micro_fixation, histogram, angle_among_3consecutive_Points, \
    angle_btw_2consecutive_points_in_vector, ngram_features, ngram_bins
import schau_mir_in_die_augen.datasets.dataset_helpers as helpers
from schau_mir_in_die_augen.trajectory_split import ivt

memory = Memory('/tmp/smida-cache', verbose=0)
#memory.clear(warn=False)


def trajectory_split_prepare_ivt_cached(xy, ds, vel_threshold=50, min_fix_duration=.1):
    return memory.cache(trajectory_split_prepare_ivt)(xy, ds.sample_rate, ds.get_screen_params(), vel_threshold, min_fix_duration)


def trajectory_split_prepare_ivt(xy, sample_rate, screen_params, vel_threshold=50, min_fix_duration=.1):
    """ Generate feature vectors for all saccades and fixations in a trajectory

    :param xy: ndarray
        2D array of gaze points (x,y)
    :param label: any
         single class this trajectory belongs to
    :return: list, list
        list of feature vectors (each is a 1D ndarray) and list of classes(these will all be the same)
    """

    angle = helpers.convert_pixel_coordinates_to_angles(xy, *screen_params.values())
    smoothed_angle = np.asarray(helpers.savgol_filter_trajectory(angle)).T
    smoothed_vel_xy = calculate_velocity(smoothed_angle, sampleRate=sample_rate)
    smoothed_vel = np.linalg.norm(smoothed_vel_xy, axis=1)
    smoothed_pixels = helpers.convert_angles_to_pixel_coordinates(smoothed_angle, *screen_params.values())

    # threshold and duration from the paper
    sac, fix = ivt(smoothed_vel, vel_threshold=vel_threshold, min_fix_duration=min_fix_duration, sampleRate=sample_rate)
    if len(sac) == 0 or len(fix) == 0:
        print("empty")

    features = pd.DataFrame()
    prev_part = slice(0,0)
    for is_saccade, windows in zip([True, False], [sac, fix]):
        for s in windows:
            # some features need at least 4 points
            if s[1] - s[0] < 4:
                continue
            # slice expects start and stop, where stop is exclusive
            part = slice(s[0], s[1] + 1)
            all_feats = all_features(smoothed_pixels[part, :], sample_rate, screen_params.values(),
                                     prev_xy=smoothed_pixels[prev_part, :], omit_our=True, omit_stats=False)
            all_feats['is_saccade'] = is_saccade
            features = features.append(all_feats, sort=False)
            prev_part = part

    return features


def all_features(xy, sample_rate, screen_params, prev_xy=np.array([]), omit_stats=False, omit_our=True):
    """ Calculate all features for a given trajectory.

    Input:
    xy: ndarray
        2D array of gaze points (x,y)
    sampleRate: float
        Sample rate of tracker in Hz
    xy_prev: ndarray
        2D array of last saccade's gaze points (x,y)

    Returns: dataframe
        1D list of numerical features with feature names in table header

    """
    assert len(xy.shape) == 2

    duration = xy.shape[0] / sample_rate  # calculate each saccadic duration

    # features in screen space
    total_distance = total_length(calculate_distance_vector(xy))
    win_angle = angle_between_first_and_last_points(xy)
    vel = calculate_velocity(xy, sample_rate)
    vel_x, vel_y = vel[:, 0], vel[:, 1]
    acc_x = angular_acceleration(vel_x, sample_rate)
    acc_y = angular_acceleration(vel_y, sample_rate)

    # angular features
    xy_angles = dataset_helpers.convert_pixel_coordinates_to_angles(xy, *screen_params)
    angular_vel = np.linalg.norm(calculate_velocity(xy_angles, sample_rate), axis=1)
    angular_acc = angular_acceleration(angular_vel, sample_rate)

    if len(prev_xy) > 0:
        angle_with_previous_sacc = win_angle - angle_between_first_and_last_points(prev_xy)
        distance_from_previous_sacc = distance_from_previous_fix_or_sacc(xy, prev_xy)
    else:
        angle_with_previous_sacc = 0
        distance_from_previous_sacc = 0

    features = {'duration': duration,
                'std_x': np.std(xy[:, 0]),
                'std_y': np.std(xy[:, 1]),
                'path_len': total_distance,
                'angle_prev_win': angle_with_previous_sacc,
                'dist_prev_win': distance_from_previous_sacc,
                'skew_x': skew(xy[:, 0]),
                'skew_y': skew(xy[:, 1]),
                'kurtosis_x': kurtosis(xy[:, 0]),
                'kurtosis_y': kurtosis(xy[:, 1]),
                'dispersion': calculate_dispersion(xy),
                'avg_vel': total_distance / duration,
                'win_ratio': np.max(angular_vel) / duration,
                'win_angle': win_angle,
                'win_amplitude': distance_between_first_and_last_points(xy)}
    if not omit_stats:
        features = {**features,
                    **{**dict(zip(stat_names('vel_x', True), statistics(vel_x))),
                       **dict(zip(stat_names('vel_y', True), statistics(vel_y))),
                       **dict(zip(stat_names('acc_x', True), statistics(acc_x))),
                       **dict(zip(stat_names('acc_y', True), statistics(acc_y))),
                       **dict(zip(stat_names('ang_vel', True), statistics(angular_vel))),
                       **dict(zip(stat_names('ang_acc', True), statistics(angular_acc)))}}

    if not omit_our:
        # TODO we could split this for x and y -> make this a function
        angular_vel_total = np.sum(angular_vel)
        angular_acc_total = np.sum(angular_acc)
        angular_acc_pos = angular_acc[angular_acc > 0]
        acc_pos_max = np.max(angular_acc_pos) if len(angular_acc_pos) else 0
        acc_pos_min = np.min(angular_acc_pos) if len(angular_acc_pos) else 0
        angular_acc_pos_diff = acc_pos_max - acc_pos_min
        angular_acc_pos_factor = acc_pos_max / acc_pos_min if acc_pos_min != 0 else 0

        angular_acc_neg = angular_acc[angular_acc < 0]
        acc_neg_max = np.max(angular_acc_neg) if len(angular_acc_neg) else 0
        acc_neg_min = np.min(angular_acc_neg) if len(angular_acc_neg) else 0
        angular_acc_neg_diff = acc_neg_max - acc_neg_min
        angular_acc_neg_factor = acc_pos_max / acc_neg_min if acc_neg_min != 0 else 0

        micro_fix_ = micro_fixation(xy, name_prefix='thresh_none')
        micro_fix_5 = micro_fixation(xy, 5, name_prefix='thresh_5')
        micro_fix_10 = micro_fixation(xy, 10, name_prefix='thresh_10')
        histogram_steps = 20
        angle_3points_list = angle_among_3consecutive_Points(xy)
        histogram_features = histogram(np.vstack(angle_3points_list), histogram_steps)

        # n-gram features
        ngram_directions_number = 8
        angle_2points_list = angle_btw_2consecutive_points_in_vector(xy)
        uni_ngram = ngram_features(ngram_bins(angle_2points_list, 8), 1, name_prefix='unigram')
        bi_ngram = ngram_features(ngram_bins(angle_2points_list, ngram_directions_number), 2, name_prefix='bigram')

        features = {**features,
            **{**dict(zip(acceleration_features_names("deacc"), acceleration_features(angular_acc[angular_acc < 0]))),
               **dict(zip(acceleration_features_names("acc"), acceleration_features(angular_acc[angular_acc > 0]))),
               **dict(zip(direction_changes_names(), direction_changes(xy))),
               'angular_vel_total': angular_vel_total,
               'angular_acc_neg_diff': angular_acc_neg_diff,
               'angular_acc_neg_factor': angular_acc_neg_factor,
               'angular_acc_pos_diff': angular_acc_pos_diff,
               'angular_acc_pos_factor': angular_acc_pos_factor,
               'angular_acc_total': angular_acc_total,
               **dict(zip(uni_ngram[1], uni_ngram[0])),
               **dict(zip(bi_ngram[1], bi_ngram[0])),
               **dict(zip(histogram_features[1], histogram_features[0])),
               **dict(zip(micro_fix_[1], micro_fix_[0])),
               **dict(zip(micro_fix_5[1], micro_fix_5[0])),
               **dict(zip(micro_fix_10[1], micro_fix_10[0])),
            }
        }

    return pd.DataFrame(features, index=[0])

def sac_subset(feats, text=True):
    if text:
        fix_cols = ['dispersion'] \
                   + stat_names('ang_vel', False)[1:] \
                   + stat_names('ang_acc', False)[:-1] \
                   + ['std_x',
                      'std_y',
                      'path_len',
                      'angle_prev_win', 'dist_prev_win', 'win_ratio', 'win_angle', 'win_amplitude'] \
                   + stat_names('vel_x', False) \
                   + stat_names('vel_y', False) \
                   + stat_names('acc_x', False) \
                   + stat_names('acc_y', False)
    else:
        # RAN data
        vel_names = stat_names('vel_y', False)
        acc_names = stat_names('acc_y', False)
        fix_cols = ['dispersion'] \
                   + stat_names('ang_vel', False)[3:] \
                   + stat_names('ang_acc', False) \
                   + ['std_x',
                      'std_y',
                      'path_len',
                      'angle_prev_win', 'dist_prev_win', 'win_ratio', 'win_angle', 'win_amplitude'] \
                   + stat_names('vel_x', False) \
                   + vel_names[:4] + [vel_names[5]] \
                   + stat_names('acc_x', False) \
                   + acc_names[:2] + acc_names[3:]

    return feats[fix_cols]


def fix_subset(feats, text=True):
    if text:
        fix_cols = ['std_y',
                    'path_len',
                    'angle_prev_win',
                    'dist_prev_win',
                    'skew_x',
                    'skew_y',
                    'kurtosis_y',
                    'dispersion',
                    'avg_vel']
    else:
        # RAN data
        fix_cols = ['duration',
                    'path_len',
                    'angle_prev_win',
                    'dist_prev_win',
                    'skew_x',
                    'skew_y',
                    'kurtosis_y',
                    'dispersion',
                    'avg_vel']

    return feats[fix_cols]


def paper_all_subset(feats):
    fix_cols = ['duration',
                'path_len',
                'skew_x',
                'skew_y',
                'kurtosis_x',
                'kurtosis_y',
                'avg_vel',
                'std_x',
                'std_y',
                'path_len',
                'angle_prev_win',
                'dist_prev_win',
                'win_ratio',
                'win_angle',
                'win_amplitude',
                'dispersion'] \
               + stat_names('ang_vel', False) \
               + stat_names('ang_acc', False) \
               + stat_names('vel_x', False) \
               + stat_names('vel_y', False) \
               + stat_names('acc_x', False) \
               + stat_names('acc_y', False)

    return feats[fix_cols]


def trajectory_features(xy, sample_rate, screen_params):
    general_gaze = general_gaze_points_features(xy, sample_rate, screen_params)

    features = {**dict(zip(general_gaze[1], general_gaze[0]))}
    return pd.DataFrame(features, index=[0])
