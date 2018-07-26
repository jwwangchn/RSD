# -*- coding: utf-8 -*_
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import time

from src.utils.util import get_iou_matrix_tf


if __name__ == '__main__':
    # sess = tf.InteractiveSession()
    coord = np.array([[150, 150, 50, 100, -90],
                      [150, 150, 100, 50, -90],
                      [150, 150, 50, 100, -45],
                      [150, 150, 100, 50, -45]])

    coord1 = np.array([150, 150, 100, 50, 0])
    start = time.time()
    for idx in range(100):
        iou = get_iou_matrix_tf(coord, coord1, use_gpu=False)
    end = time.time()
    print("CPU Time: ", end - start)

    start = time.time()
    for idx in range(100):
        iou = get_iou_matrix_tf(coord, coord1, use_gpu=True)
    end = time.time()
    print("GPU Time: ", end - start)

    print(iou)
    print(np.argsort(iou)[::-1])