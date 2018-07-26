# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Model configuration for pascal dataset"""

import numpy as np

from config import base_model_config


def kitti_squeezeDet_config():
    """Specify the parameters to tune below."""
    mc = base_model_config('KITTI')

    mc.IMAGE_WIDTH = 342
    mc.IMAGE_HEIGHT = 342
    mc.BATCH_SIZE = 20

    mc.WEIGHT_DECAY = 0.0001
    mc.LEARNING_RATE = 0.01
    mc.DECAY_STEPS = 10000
    mc.MAX_GRAD_NORM = 1.0
    mc.MOMENTUM = 0.9
    mc.LR_DECAY_FACTOR = 0.5

    mc.LOSS_COEF_BBOX = 5.0
    mc.LOSS_COEF_CONF_POS = 75.0
    mc.LOSS_COEF_CONF_NEG = 100.0
    mc.LOSS_COEF_CLASS = 1.0

    mc.PLOT_PROB_THRESH = 0.4
    mc.NMS_THRESH = 0.4
    mc.PROB_THRESH = 0.005
    mc.TOP_N_DETECTION = 64

    mc.DATA_AUGMENTATION = False
    mc.DRIFT_X = 150
    mc.DRIFT_Y = 100
    mc.EXCLUDE_HARD_EXAMPLES = False

    mc.ANCHOR_BOX = set_anchors(mc)
    mc.ANCHORS = len(mc.ANCHOR_BOX)
    mc.ANCHOR_PER_GRID = 81

    return mc


def set_anchors(mc):
    H, W, B = 21, 21, 9
    anchor_base = np.reshape([np.array(
                                        [[36., 37.], 
                                        [366., 174.], 
                                        [115., 59.],
                                        [162., 87.], 
                                        [38., 90.], 
                                        [258., 173.],
                                        [224., 108.], 
                                        [78., 170.], 
                                        [72., 43.]])] * H * W, (H, W, B, 2))
    angle_base = np.reshape([np.array(
                                        [[0.0],
                                        [20.0],
                                        [40.0],
                                        [60.0],
                                        [80.0],
                                        [100.0],
                                        [120.0],
                                        [140.0],
                                        [160.0]]
                                        )] * H * W, (H, W, B, 1))
    center_x = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, W + 1) * float(mc.IMAGE_WIDTH) / (W + 1)] * H * B),
                (B, H, W)
            ),
            (1, 2, 0)
        ),
        (H, W, B, 1)
    )
    
    center_y = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, H + 1) * float(mc.IMAGE_HEIGHT) / (H + 1)] * W * B),
                (B, W, H)
            ),
            (2, 1, 0)
        ),
        (H, W, B, 1)
    )
    anchors = np.reshape(
        np.concatenate((center_x, center_y, anchor_base, angle_base), axis=3),
        (-1, 5)
    )
    return anchors

if __name__ == '__main__':
    mc = kitti_squeezeDet_config()
    anchors = set_anchors(mc)
    print len(anchors), anchors.shape
    print anchors[0:1:1, :]