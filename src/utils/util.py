# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Utility functions."""

import numpy as np
import time
import tensorflow as tf

from src.utils.iou_cpu import get_iou_matrix
from src.utils.nms_gpu import rotate_gpu_nms
from src.utils.iou_gpu import rbbx_overlaps

def iou(box1, box2):
    """Compute the Intersection-Over-Union of two given boxes.

    Args:
      box1: array of 4 elements [cx, cy, width, height].
      box2: same as above
    Returns:
      iou: a float number in range [0, 1]. iou of the two boxes.
    """

    lr = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
         max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    if lr > 0:
        tb = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
             max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb > 0:
            intersection = tb * lr
            union = box1[2] * box1[3] + box2[2] * box2[3] - intersection

            return intersection / union

    return 0


def batch_iou(boxes, box):
    """Compute the Intersection-Over-Union of a batch of boxes with another
    box.

    Args:
      box1: 2D array of [cx, cy, width, height].
      box2: a single array of [cx, cy, width, height]
    Returns:
      ious: array of a float number in range [0, 1].
    """
    lr = np.maximum(
        np.minimum(boxes[:, 0] + 0.5 * boxes[:, 2], box[0] + 0.5 * box[2]) - \
        np.maximum(boxes[:, 0] - 0.5 * boxes[:, 2], box[0] - 0.5 * box[2]),
        0
    )
    tb = np.maximum(
        np.minimum(boxes[:, 1] + 0.5 * boxes[:, 3], box[1] + 0.5 * box[3]) - \
        np.maximum(boxes[:, 1] - 0.5 * boxes[:, 3], box[1] - 0.5 * box[3]),
        0
    )
    inter = lr * tb
    union = boxes[:, 2] * boxes[:, 3] + box[2] * box[3] - inter
    return inter / union


def nms(boxes, probs, threshold):
    """Non-Maximum supression.
    Args:
      boxes: array of [cx, cy, w, h] (center format)
      probs: array of probabilities
      threshold: two boxes are considered overlapping if their IOU is largher than
          this threshold
      form: 'center' or 'diagonal'
    Returns:
      keep: array of True or False.
    """

    order = probs.argsort()[::-1]
    keep = [True] * len(order)

    for i in range(len(order) - 1):
        ovps = batch_iou(boxes[order[i + 1:]], boxes[order[i]])
        for j, ov in enumerate(ovps):
            if ov > threshold:
                keep[order[j + i + 1]] = False
    return keep

def py_cpu_nms(dets, probs, thresh):  
    """Pure Python NMS baseline."""
    # [cx, cy, w, h]
    x1 = dets[:, 0] - dets[:, 2]/2.0
    y1 = dets[:, 1] - dets[:, 3]/2.0
    x2 = dets[:, 0] + dets[:, 2]/2.0
    y2 = dets[:, 1] + dets[:, 3]/2.0
  
    areas = dets[:, 2] * dets[:, 3]
    order = probs.argsort()[::-1]
    keep = []
    keep_ = [False]*len(order)
    while order.size > 0:
        i = order[0]
        # keep.append(i)
        keep_[i] = True
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
  
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr < thresh)[0]
        order = order[inds + 1]
    return keep_


# TODO(bichen): this is not equivalent with full NMS. Need to improve it.
def recursive_nms(boxes, probs, threshold, form='center'):
    """Recursive Non-Maximum supression.
    Args:
      boxes: array of [cx, cy, w, h] (center format) or [xmin, ymin, xmax, ymax]
      probs: array of probabilities
      threshold: two boxes are considered overlapping if their IOU is largher than
          this threshold
      form: 'center' or 'diagonal'
    Returns:
      keep: array of True or False.
    """

    assert form == 'center' or form == 'diagonal', \
        'bounding box format not accepted: {}.'.format(form)

    if form == 'center':
        # convert to diagonal format
        boxes = np.array([bbox_transform(b) for b in boxes])

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    hidx = boxes[:, 0].argsort()
    keep = [True] * len(hidx)

    def _nms(hidx):
        order = probs[hidx].argsort()[::-1]

        for idx in range(len(order)):
            if not keep[hidx[order[idx]]]:
                continue
            xx2 = boxes[hidx[order[idx]], 2]
            for jdx in range(idx + 1, len(order)):
                if not keep[hidx[order[jdx]]]:
                    continue
                xx1 = boxes[hidx[order[jdx]], 0]
                if xx2 < xx1:
                    break
                w = xx2 - xx1
                yy1 = max(boxes[hidx[order[idx]], 1], boxes[hidx[order[jdx]], 1])
                yy2 = min(boxes[hidx[order[idx]], 3], boxes[hidx[order[jdx]], 3])
                if yy2 <= yy1:
                    continue
                h = yy2 - yy1
                inter = w * h
                iou = inter / (areas[hidx[order[idx]]] + areas[hidx[order[jdx]]] - inter)
                if iou > threshold:
                    keep[hidx[order[jdx]]] = False

    def _recur(hidx):
        if len(hidx) <= 20:
            _nms(hidx)
        else:
            mid = len(hidx) / 2
            _recur(hidx[:mid])
            _recur(hidx[mid:])
            _nms([idx for idx in hidx if keep[idx]])

    _recur(hidx)

    return keep


def sparse_to_dense(sp_indices, output_shape, values, default_value=0):
    """Build a dense matrix from sparse representations.

    Args:
      sp_indices: A [0-2]-D array that contains the index to place values.
      shape: shape of the dense matrix.
      values: A {0,1}-D array where values corresponds to the index in each row of
      sp_indices.
      default_value: values to set for indices not specified in sp_indices.
    Return:
      A dense numpy N-D array with shape output_shape.
    """

    assert len(sp_indices) == len(values), \
        'Length of sp_indices is not equal to length of values'

    array = np.ones(output_shape) * default_value
    for idx, value in zip(sp_indices, values):
        array[tuple(idx)] = value
    return array


def bgr_to_rgb(ims):
    """Convert a list of images from BGR format to RGB format."""
    out = []
    for im in ims:
        out.append(im[:, :, ::-1])
    return out


def bbox_transform(bbox):
    """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
    for numpy array or list of tensors.
    """
    with tf.variable_scope('bbox_transform') as scope:
        cx, cy, w, h = bbox
        out_box = [[]] * 4
        out_box[0] = cx - w / 2
        out_box[1] = cy - h / 2
        out_box[2] = cx + w / 2
        out_box[3] = cy + h / 2

    return out_box


def bbox_transform_inv(bbox):
    """convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]. Works
    for numpy array or list of tensors.
    """
    with tf.variable_scope('bbox_transform_inv') as scope:
        xmin, ymin, xmax, ymax = bbox
        out_box = [[]] * 4

        width = xmax - xmin + 1.0
        height = ymax - ymin + 1.0
        out_box[0] = xmin + 0.5 * width
        out_box[1] = ymin + 0.5 * height
        out_box[2] = width
        out_box[3] = height

    return out_box


class Timer(object):
    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.duration = 0.0
        self.average_time = 0.0

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.duration = time.time() - self.start_time
        self.total_time += self.duration
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.duration


def safe_exp(w, thresh):
    """Safe exponential function for tensors."""

    slope = np.exp(thresh)
    with tf.variable_scope('safe_exponential'):
        lin_bool = w > thresh
        lin_region = tf.to_float(lin_bool)

        lin_out = slope * (w - thresh + 1.)
        exp_out = tf.exp(tf.where(lin_bool, tf.zeros_like(w), w))

        out = lin_region * lin_out + (1. - lin_region) * exp_out
    return out







def get_iou_matrix_tf(boxes1, boxes2, use_gpu=True, gpu_id=0):
    '''

    :param boxes_list1:[N, 5] numpy.array
    :param boxes_list2: [M, 5] numpy.array
    :return: [N, M] numpy.array
    '''
    if boxes1.ndim == 1:
        boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1:
        boxes2 = np.expand_dims(boxes2, axis=0)

    if use_gpu:
        boxes1 = boxes1.astype(np.float32)
        boxes2 = boxes2.astype(np.float32)
        iou_matrix = rbbx_overlaps(boxes1, boxes2, gpu_id).astype(np.float32)
    else:
        boxes1 = boxes1.astype(np.float32)
        boxes2 = boxes2.astype(np.float32)
        iou_matrix = get_iou_matrix(boxes1, boxes2).astype(np.float32)

    iou_matrix = np.reshape(iou_matrix, (boxes1.shape[0], boxes2.shape[0]))

    return iou_matrix


def nms_rotate_tf(boxes_list, scores, iou_threshold, max_output_size, use_gpu=True, gpu_id=0):

    if use_gpu:
        det_tensor = tf.concat([boxes_list, tf.expand_dims(scores, axis=1)], axis=1)
        keep = tf.py_func(rotate_gpu_nms,
                          inp=[det_tensor, iou_threshold, gpu_id],
                          Tout=tf.int64)
        keep = tf.cond(
            tf.greater(tf.shape(keep)[0], max_output_size),
            true_fn=lambda: tf.slice(keep, [0], [max_output_size]),
            false_fn=lambda: keep)
        keep = tf.reshape(keep, [-1])
        return keep
    else:
        raise NotImplementedError("not implemented the CPU vesion because of low speed")

def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):

    keep = []

    order = scores.argsort()[::-1]
    num = boxes.shape[0]

    suppressed = np.zeros((num), dtype=np.int)

    for _i in range(num):
        if len(keep) >= max_output_size:
            break
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + cfgs.EPSILON)

            if inter >= iou_threshold:
                suppressed[j] = 1

    return np.array(keep, np.int64)


#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------


def get_mask_tf(rotate_rects, featuremap_size):
    mask_tensor = tf.py_func(get_mask,
                            inp=[rotate_rects, featuremap_size],
                            Tout=tf.float32)
    mask_tensor = tf.reshape(mask_tensor, [tf.shape(rotate_rects)[0], featuremap_size, featuremap_size]) # [300, 14, 14]

    return mask_tensor


def get_mask(rotate_rects, featuremap_size):

    all_mask = []
    for a_rect in rotate_rects:
        rect = ((a_rect[1], a_rect[0]), (a_rect[3], a_rect[2]), a_rect[-1])  # in tf. [x, y, w, h, theta]
        rect_eight = cv2.boxPoints(rect)
        x_list = rect_eight[:, 0:1]
        y_list = rect_eight[:, 1:2]
        min_x, max_x = np.min(x_list), np.max(x_list)
        min_y, max_y = np.min(y_list), np.max(y_list)
        x_list = x_list - min_x
        y_list = y_list - min_y

        new_rect = np.hstack([x_list*featuremap_size*1.0/(max_x-min_x+1),
                             y_list * featuremap_size * 1.0 / (max_y - min_y + 1)])
        mask_array = np.zeros([featuremap_size, featuremap_size], dtype=np.float32)
        for x in range(featuremap_size):
            for y in range(featuremap_size):
                inner_rect = cv2.pointPolygonTest(contour=new_rect, pt=(x, y), measureDist=False)
                mask_array[y, x] = np.float32(0) if inner_rect == -1 else np.float32(1)
        all_mask.append(mask_array)
    return np.array(all_mask)