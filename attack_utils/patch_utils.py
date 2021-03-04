from misc_utils.geometry import *
import numpy as np


def extract_roi(detections, class_id, img_bbox, min_size, patch_size):
    roi_candidates = []
    for did, detection in enumerate(detections):
        if int(detection[0]) != class_id:
            continue
        obj_bbox = tuple(map(int, detection[-4:]))
        pat_bbox = (obj_bbox[0] + (obj_bbox[2] - obj_bbox[0]) // 2 - patch_size[1] // 2,
                    obj_bbox[1] + (obj_bbox[3] - obj_bbox[1]) // 2 - patch_size[0] // 2,
                    obj_bbox[0] + (obj_bbox[2] - obj_bbox[0]) // 2 + patch_size[1] // 2,
                    obj_bbox[1] + (obj_bbox[3] - obj_bbox[1]) // 2 + patch_size[0] // 2)
        if bb_size(obj_bbox) >= min_size and bb_inside(pat_bbox, img_bbox):
            roi_candidates.append((float(detection[1]), obj_bbox, pat_bbox, did))

    # Filter out rois with overlapped patches based on non-maximum suppression
    roi_candidates = sorted(roi_candidates, key=lambda x: -x[0])
    rois = []
    for roi_candidate in roi_candidates:
        if not np.any([bb_overlap(roi_candidate[2], roi[2]) for roi in rois]):
            rois.append(roi_candidate)
    return rois


def evaluate_vanishing_patch(source_class, rois, detections_adv, detections_rand, iou_thresh=0.50):
    score_adv, score_rand = 0, 0
    for _, roi_obj_bbox, _, _ in rois:
        found = False
        for detection in detections_adv:
            det_obj_bbox = tuple(map(float, detection[-4:]))
            det_obj_class = int(detection[0])
            if det_obj_class == source_class and bb_intersection_over_union(roi_obj_bbox, det_obj_bbox) > iou_thresh:
                found = True
                break
        score_adv += (not found)

        found = False
        for detection in detections_rand:
            det_obj_bbox = tuple(map(float, detection[-4:]))
            det_obj_class = int(detection[0])
            if det_obj_class == source_class and bb_intersection_over_union(roi_obj_bbox, det_obj_bbox) > iou_thresh:
                found = True
                break
        score_rand += (not found)
    return score_adv, score_rand


def evaluate_mislabeling_patch(target_class, rois, detections_adv, detections_rand, iou_thresh=0.50):
    score_adv, score_rand = 0, 0
    for _, roi_obj_bbox, _, _ in rois:
        found = False
        for detection in detections_adv:
            det_obj_bbox = tuple(map(float, detection[-4:]))
            det_obj_class = int(detection[0])
            if det_obj_class == target_class and bb_intersection_over_union(roi_obj_bbox, det_obj_bbox) > iou_thresh:
                found = True
                break
        score_adv += found

        found = False
        for detection in detections_rand:
            det_obj_bbox = tuple(map(float, detection[-4:]))
            det_obj_class = int(detection[0])
            if det_obj_class == target_class and bb_intersection_over_union(roi_obj_bbox, det_obj_bbox) > iou_thresh:
                found = True
                break
        score_rand += found
    return score_adv, score_rand
