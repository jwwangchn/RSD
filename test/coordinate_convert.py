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
    height, width = 500, 500
    img = 255 * np.ones([height+500, width+500, 3])

    bboxes = np.array([[500, 500, 500, 500, -10]])    # (cx, cy, w, h, angle)
    print "bboxes: ", bboxes

    coordinates = forward_convert(bboxes, False)    # (x1, y1, x2, y2, x3, y3, x4, y4)
    print "coordinates: ", coordinates
    # print("xy (min max) bboxes: ", coordinates)

    x1 = np.minimum(np.maximum(0, coordinates[:, 0]), width - 1.0)
    x2 = np.minimum(np.maximum(0, coordinates[:, 1]), height - 1.0)
    x3 = np.minimum(np.maximum(0, coordinates[:, 2]), width - 1.0)
    x4 = np.minimum(np.maximum(0, coordinates[:, 3]), height - 1.0)
    x5 = np.minimum(np.maximum(0, coordinates[:, 4]), width - 1.0)
    x6 = np.minimum(np.maximum(0, coordinates[:, 5]), height - 1.0)
    x7 = np.minimum(np.maximum(0, coordinates[:, 6]), width - 1.0)
    x8 = np.minimum(np.maximum(0, coordinates[:, 7]), height - 1.0)

    new_coordinates = np.reshape(np.array([x1, x2, x3, x4, x5, x6, x7, x8]), (-1, 8))
    print "new_coordinates: ", new_coordinates

    new_bboxes = back_forward_convert(new_coordinates, False)
    print "new_bboxes: ", new_bboxes


    for i, bbox in enumerate(bboxes):
        rect = ((bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox[4])
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        cv2.drawContours(img, [rect], -1, (0, 255, 0), 3)   # original rect

        print rect
        coordinates_reshape = coordinates.reshape((4, 2))
        print coordinates_reshape
        cv2.drawContours(img, [np.int0(coordinates_reshape)], -1, (255, 0, 0), 3)   # original rect


        rect = ((new_bboxes[i, 0], new_bboxes[i, 1]), (new_bboxes[i, 2], new_bboxes[i, 3]), new_bboxes[i, 4] + 10)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        cv2.drawContours(img, [rect], -1, (0, 0, 255), 3)   # cut rect

        new_coordinates_reshape = new_coordinates.reshape((4, 2))
        cv2.drawContours(img, [np.int0(new_coordinates_reshape)], -1, (255, 255, 0), 3)

    cv2.line(img,(0, 500),(500,500), (255,0,0), 5)
    cv2.line(img,(500, 0),(500,500), (255,0,0), 5)


    cv2.imshow("img ", img)
    cv2.waitKey()
    cv2.destroyAllWindows()