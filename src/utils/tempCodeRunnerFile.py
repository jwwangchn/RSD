x1 = (xy_bboxes[:, 0] > 0) & (xy_bboxes[:, 0] < width)
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