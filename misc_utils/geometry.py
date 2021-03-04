def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection area and dividing it by the sum
    # of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def bb_size(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def bb_inside(boxA, boxB):  # return True if boxA is inside boxB
    return boxA[0] >= boxB[0] and boxA[1] >= boxB[1] and boxA[2] <= boxB[2] and boxA[3] <= boxB[3]


def bb_overlap(boxA, boxB):
    return bb_intersection_over_union(boxA, boxB) > 0
