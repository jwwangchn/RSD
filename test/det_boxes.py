# # -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
 
bboxes = np.ones((10, 4))
det_boxes = tf.transpose(tf.stack([bboxes]), (1, 2, 0))

sess = tf.InteractiveSession()
det_boxes = sess.run(det_boxes)

print(det_boxes.shape)