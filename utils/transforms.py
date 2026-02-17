import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix


def get_sensor_to_global(nusc, sample_data_token):
    # return the 4x4 matrix transfomation for sensor local -> global world

    sd = nusc.get('sample_data', sample_data_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])

    T_s2e = transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=False)
    T_e2g = transform_matrix(ep['translation'], Quaternion(ep['rotation']), inverse=False)

    return np.dot(T_e2g, T_s2e)


def transform_points_sensor_to_sensor(nusc, src_sd_token, dst_sd_token, points):
    # return the 4x4 matrix transfomation for sensor local -> destination sensor frame

    if points.shape[0] != 3:
        raise ValueError('points must be shape (3, N)')

    T_src2g = get_sensor_to_global(nusc, src_sd_token)
    T_dst2g = get_sensor_to_global(nusc, dst_sd_token)

    # src -> global -> dst
    T_src2dst = np.linalg.inv(T_dst2g).dot(T_src2g)

    n = points.shape[1]
    homog = np.vstack((points, np.ones((1, n))))
    transformed = T_src2dst.dot(homog)

    return transformed[:3, :]


def get_camera_intrinsic(nusc, cam_sd_token):
    sd = nusc.get('sample_data', cam_sd_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    K = np.array(cs['camera_intrinsic'])
    return K
