# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


def forward_convert(coordinate, with_label=False):
    """
    :param coordinate: format [x_c, y_c, w, h, theta]
    :return: format [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    boxes = []
    if with_label:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], rect[5]])
    else:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            boxes.append(np.reshape(box, [-1, ]))

    return np.array(boxes, dtype=np.float32)


def back_forward_convert(coordinates, with_label=True):
    """
    :param coordinates: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)]
    :param with_label: default True
    :return: format [x_c, y_c, w, h, theta, (label)]
    range of theta is  [-90, 0)
    """

    boxes = []
    if with_label:
        for rect in coordinates:
            box = np.int0(rect[:-1])
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([x, y, w, h, theta, rect[-1]])
    else:
        for rect in coordinates:
            box = np.int0(rect)
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([x, y, w, h, theta])

    return np.array(boxes, dtype=np.float32)



if __name__ == '__main__':
    coord = np.array([[150, 150, 50, 100, -90],
                      [150, 150, 100, 50, -90],
                      [150, 150, 50, 100, -45],
                      [150, 150, 100, 50, -45]])

    coord1 = np.array([[100, 100, 50, 50, -10, 1],
                      [150, 150, 100, 50, -90, 1],
                      [150, 150, 100, 50, 45, 1],
                      [150, 150, 100, 50, -45, 1]])

    coord1 = np.array([[100, 100, 50, 50, -10, 1],
                        [100, 100, 50, 50, -10, 1]])

    xy_bboxes = forward_convert(coord1, True)
    print("xy (min max) bboxes: ", xy_bboxes)

    height, width = 100, 100

    x1 = np.minimum(np.maximum(0, xy_bboxes[:, 0]), width - 1.0)
    x2 = np.minimum(np.maximum(0, xy_bboxes[:, 1]), height - 1.0)
    x3 = np.minimum(np.maximum(0, xy_bboxes[:, 2]), width - 1.0)
    x4 = np.minimum(np.maximum(0, xy_bboxes[:, 3]), height - 1.0)
    x5 = np.minimum(np.maximum(0, xy_bboxes[:, 4]), width - 1.0)
    x6 = np.minimum(np.maximum(0, xy_bboxes[:, 5]), height - 1.0)
    x7 = np.minimum(np.maximum(0, xy_bboxes[:, 6]), width - 1.0)
    x8 = np.minimum(np.maximum(0, xy_bboxes[:, 7]), height - 1.0)
    print(x1, x2, x3, x4, x5, x6, x7, x8)

    print(np.array([x1, x2, x3, x4, x5, x6, x7, x8]))
    back_forward_convert(np.array([x1, x2, x3, x4, x5, x6, x7, x8]))

    # x1 = (xy_bboxes[:, 0] > 0) & (xy_bboxes[:, 0] < width)
    # y1 = (xy_bboxes[:, 1] > 0) & (xy_bboxes[:, 1] < height)
    # x2 = (xy_bboxes[:, 2] > 0) & (xy_bboxes[:, 2] < width)
    # y2 = (xy_bboxes[:, 3] > 0) & (xy_bboxes[:, 3] < height)
    # x3 = (xy_bboxes[:, 4] > 0) & (xy_bboxes[:, 4] < width)
    # y3 = (xy_bboxes[:, 5] > 0) & (xy_bboxes[:, 5] < height)
    # x4 = (xy_bboxes[:, 6] > 0) & (xy_bboxes[:, 6] < width)
    # y4 = (xy_bboxes[:, 7] > 0) & (xy_bboxes[:, 7] < height)

    # delete_flag = x1 & y1 & x2 & y2 & x3 & y3 & x4 & y4

    # xy_bboxes = xy_bboxes[delete_flag, ...]
    # print(xy_bboxes)
    # xytheta_bboxes = back_forward_convert(xy_bboxes, with_label=True)

    # print(delete_flag)
    # print(xytheta_bboxes)



    # coord2 = np.array([[60, 20, 71, 79, 120, 71, 128, 120, 1.]])

    # det_coord2 = back_forward_convert(coord2)
    # print(det_coord2)
    # coord2 = forward_convert(det_coord2)
    # print(coord2)

