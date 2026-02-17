import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box, transform_matrix


def get_dynamic_boxes(nusc, sample):
    # return boxes for likely-moving categories
    ann_tokens = sample.get('anns', sample.get('annotations', []))
    boxes = []
    prefixes = ['vehicle', 'human', 'bicycle', 'motorcycle'] # using prefixes so that we get vehicle.car and vehicle.bus etc

    for t in ann_tokens:
        a = nusc.get('sample_annotation', t)
        cat = a['category_name']
        if any(cat.startswith(p) for p in prefixes):
            boxes.append({'translation': np.array(a['translation']),
                          'size': np.array(a['size']),
                          'rotation': np.array(a['rotation'])})
    return boxes


def mask_points_in_boxes(pts_s, boxes_g, T_s2g, buffer=0.15):
    # pts_s: (3,N) in sensor frame. boxes_g in global. T_s2g sensor->global
    N = pts_s.shape[1]
    mask = np.zeros(N, dtype=bool)
    T_g2s = np.linalg.inv(T_s2g)

    for b in boxes_g:
        # make official Box in global
        box = Box(b['translation'], b['size'], Quaternion(b['rotation']))

        # move box into sensor frame: translate then rotate
        box.translate(-T_s2g[:3, 3])
        box.rotate(Quaternion(matrix=T_g2s[:3, :3]))

        # tiny inflation to handle scan-time jitter
        box.wlh = box.wlh + buffer

        mask |= points_in_box(box, pts_s)

    return mask
