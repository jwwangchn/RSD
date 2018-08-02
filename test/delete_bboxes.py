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


def delete_outside_bbox(bboxes, height, width):
    coordinates = forward_convert(coordinate=bboxes, with_label=False)
    # [x1, y1, x2, y2, x3, y3, x4, y4]
    x1 = (coordinates[:, 0] > 0) & (coordinates[:, 0] < width)
    y1 = (coordinates[:, 1] > 0) & (coordinates[:, 1] < height)
    x2 = (coordinates[:, 2] > 0) & (coordinates[:, 2] < width)
    y2 = (coordinates[:, 3] > 0) & (coordinates[:, 3] < height)
    x3 = (coordinates[:, 4] > 0) & (coordinates[:, 4] < width)
    y3 = (coordinates[:, 5] > 0) & (coordinates[:, 5] < height)
    x4 = (coordinates[:, 6] > 0) & (coordinates[:, 6] < width)
    y4 = (coordinates[:, 7] > 0) & (coordinates[:, 7] < height)

    mask = x1 & y1 & x2 & y2 & x3 & y3 & x4 & y4
    coordinates = coordinates[mask, ...]
    bboxes = back_forward_convert(coordinates, with_label=False)

    return bboxes, mask


if __name__ == '__main__':
    bboxes = np.array([[100, 100, 50, 50, -10],
                      [150, 150, 100, 50, -90],
                      [150, 150, 100, 50, 45],
                      [150, 150, 100, 50, -45]])
    
    height, width = 500, 500

    bboxes, mask = delete_outside_bbox(bboxes, height, width)

    print(bboxes, mask)