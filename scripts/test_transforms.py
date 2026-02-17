import os
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from PIL import Image

from utils import transforms as tr


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    nusc = NuScenes(version='v1.0-mini', dataroot=os.path.join(repo_root, 'data'), verbose=False)

    sample = nusc.sample[0]
    lidar_sd_token = sample['data']['LIDAR_TOP']
    cam_sd_token = sample['data']['CAM_FRONT']

    # load cloud
    lidar_sd = nusc.get('sample_data', lidar_sd_token)
    lidar_path = os.path.join(nusc.dataroot, lidar_sd['filename'])
    pc = LidarPointCloud.from_file(lidar_path)
    points = pc.points[:3, :]

    # transform points into camera frame, using sensor specific poses
    pts_cam = tr.transform_points_sensor_to_sensor(nusc, lidar_sd_token, cam_sd_token, points)

    # project using camera intrinsics
    K = tr.get_camera_intrinsic(nusc, cam_sd_token)
    K4 = np.hstack((K, np.zeros((3, 1))))

    im_coords = view_points(pts_cam, K4, normalize=True)

    cam_sd = nusc.get('sample_data', cam_sd_token)
    im_path = os.path.join(nusc.dataroot, cam_sd['filename'])
    im = Image.open(im_path)
    w, h = im.size

    xs = im_coords[0, :]
    ys = im_coords[1, :]
    depths = pts_cam[2, :]

    mask = (depths > 0) & (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs = xs[mask]
    ys = ys[mask]
    depths = depths[mask]

    # plot and save
    plt.figure(figsize=(12, 8))
    plt.imshow(im)
    plt.scatter(xs, ys, c=depths, s=0.5, cmap='viridis')
    plt.axis('off')

    out_dir = os.path.join(repo_root, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'proj_sample0.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close()

    print('saved to:', out_path)


if __name__ == '__main__':
    main()
