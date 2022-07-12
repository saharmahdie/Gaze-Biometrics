import numpy as np

from schau_mir_in_die_augen.datasets.Bioeye import BioEye
from schau_mir_in_die_augen.datasets.rigas import RigasDataset
import schau_mir_in_die_augen.datasets.dataset_helpers as helpers
from schau_mir_in_die_augen.features import calculate_velocity, statistics, stat_names

rig_ds = RigasDataset()
rig_train = rig_ds.load_training()
bio_ds = BioEye(BioEye.Subsets.TEX_30min_dv)
bio_train = bio_ds.load_training()

def traj_stats(xy, ds):
    angle = helpers.convert_pixel_coordinates_to_angles(xy, *ds.get_screen_params().values())
    smoothed_angle = np.asarray(helpers.savgol_filter_trajectory(angle)).T
    smoothed_vel_xy = calculate_velocity(smoothed_angle, sampleRate=ds.sample_rate)
    smoothed_vel = np.linalg.norm(smoothed_vel_xy, axis=1)
    print(list(zip(stat_names(extended=True), statistics(smoothed_vel))))

for i in range(10):
    traj_stats(bio_train[0][i], bio_ds)
    traj_stats(rig_train[0][i], rig_ds)
