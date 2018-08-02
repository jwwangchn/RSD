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


def bboxes_bounding_rect(bboxes, width, height):
    coordinates = forward_convert(bboxes, False)    # (x1, y1, x2, y2, x3, y3, x4, y4)
    # for i, bbox in enumerate(bboxes):
        # draw bounding rectangle
    coordinates_reshape = coordinates.reshape((-1, 4, 2))
    bboxes = []
    for i, coordinate in enumerate(coordinates_reshape):
        x, y, w, h = cv2.boundingRect(coordinate)
        
        xmin, ymin, xmax, ymax = x, y, x + w, y + h

        xmin = np.minimum(np.maximum(0, xmin), width - 1)
        ymin = np.minimum(np.maximum(0, ymin), height - 1)
        xmax = np.maximum(np.minimum(width - 1, xmax), 0)
        ymax = np.maximum(np.minimum(height - 1, ymax), 0)
        w = xmax - xmin
        h = ymax - ymin
        cx = xmin + w/2
        cy = ymin + h/2
        bbox = [cx, cy, w, h, 0]
        bboxes.append(bbox)
    print bboxes

def 


if __name__ == '__main__':
    height, width = 500, 500
    img = 255 * np.ones([height+500, width+500, 3])

    # draw axis
    cv2.line(img,(0, 500),(500,500), (255,255,0), 5)
    cv2.line(img,(500, 0),(500,500), (255,255,0), 5)

    bboxes = np.array([[500, 500, 500, 500, -10],
                        [500, 500, 500, 500, -10]])    # (cx, cy, w, h, angle)
    
    bboxes_bounding_rect(bboxes, width, height)
    # # draw rotation bounding boxes

    # rect = ((bboxes[0, 0], bboxes[0, 1]), (bboxes[0, 2], bboxes[0, 3]), bboxes[0, 4])
    # rect = cv2.boxPoints(rect)
    # rect = np.int0(rect)
    # cv2.drawContours(img, [rect], -1, (0, 0, 255), 3)   # cut rect

    # # draw bounding rectangle
    # coordinates = forward_convert(bboxes, False)    # (x1, y1, x2, y2, x3, y3, x4, y4)
    # coordinates_reshape = coordinates.reshape((4, 2))

    # x, y, w, h = cv2.boundingRect(coordinates_reshape)
    # cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)
    
    # xmin, ymin, xmax, ymax = x, y, x + w, y + h

    # xmin = np.minimum(np.maximum(0, xmin), width - 1)
    # ymin = np.minimum(np.maximum(0, ymin), height - 1)
    # xmax = np.maximum(np.minimum(width - 1, xmax), 0)
    # ymax = np.maximum(np.minimum(height - 1, ymax), 0)
    
    # cv2.rectangle(img, (xmin-5, ymin-5), (xmax+5, ymax+5), (255, 0, 0), 3)

    # cv2.imshow("img ", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()