import os
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from PIL import Image

from utils import transforms as tr
from utils import box_utils


def main():
    r = os.path.dirname(os.path.dirname(__file__))
    n = NuScenes(version='v1.0-mini', dataroot=os.path.join(r, 'data'), verbose=False)

    s = n.sample[0]
    sd_l = s['data']['LIDAR_TOP']
    sd_c = s['data']['CAM_FRONT']

    sd = n.get('sample_data', sd_l)
    path = os.path.join(n.dataroot, sd['filename'])
    pc = LidarPointCloud.from_file(path)
    pts = pc.points[:3, :]

    # mask (sensor frame)
    T = tr.get_sensor_to_global(n, sd_l)
    boxes = box_utils.get_dynamic_boxes(n, s)
    m_dyn = box_utils.mask_points_in_boxes(pts, boxes, T)

    tot = pts.shape[1]
    nd = int(np.sum(m_dyn))
    print(f'Total points: {tot}, Dynamic: {nd}, Static: {tot-nd}')

    # projection for output
    p_cam = tr.transform_points_sensor_to_sensor(n, sd_l, sd_c, pts)
    K = tr.get_camera_intrinsic(n, sd_c)
    K4 = np.hstack((K, np.zeros((3, 1))))
    uv = view_points(p_cam, K4, normalize=True)

    sd = n.get('sample_data', sd_c)
    im = Image.open(os.path.join(n.dataroot, sd['filename']))
    w, h = im.size

    x = uv[0, :]
    y = uv[1, :]
    z = p_cam[2, :]
    vis = (z > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)

    x = x[vis]
    y = y[vis]
    dyn = m_dyn[vis]
    cols = np.where(dyn, 'red', 'lime')

    # plot and save
    plt.figure(figsize=(12, 8))
    plt.imshow(im)
    plt.scatter(x, y, c=cols, s=0.8, alpha=0.9)
    plt.axis('off')

    out = os.path.join(r, 'outputs')
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, 'segmented_sample0.png')
    plt.savefig(p, bbox_inches='tight', dpi=200)
    plt.close()
    print('Saved segmentation image to:', p)


if __name__ == '__main__':
    main()
