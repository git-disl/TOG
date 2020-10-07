import sys
import os
sys.path.append(os.path.abspath('/research/projects/robust-object-detection/frcnn_utils/pytorch-faster-rcnn/lib'))
sys.path.append(os.path.abspath('/research/projects/robust-object-detection/frcnn_utils/pytorch-faster-rcnn/tools'))
from frcnn_utils.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
from frcnn_utils.model import FRCNN_VGG16
from collections import namedtuple
from frcnn_utils import array_tool as at
from frcnn_utils.config import opt
from torch.nn import functional as F
from frcnn_utils.dataset import preprocess
from torchvision.ops import nms
from torch import nn
import numpy as np
import torch as t
import warnings
from model.bbox_transform import clip_boxes, bbox_transform_inv
from nets.vgg16 import vgg16
from model.config import cfg
warnings.simplefilter("ignore", UserWarning)

LossTuple = namedtuple('LossTuple', ['rpn_loc_loss', 'rpn_cls_loss', 'roi_loc_loss', 'roi_cls_loss', 'total_loss',
                                     'object_untargeted_loss',
                                     'object_vanishing_loss', 'object_fabrication_loss', 'object_mislabeling_loss'])


class FRCNN_RAP:
    """
    face detector wrapper
    """

    classes = ['none', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']

    def __init__(self, weights, model_img_size=(600, 600)):
        # load network
        self.model_img_size = model_img_size
        self.confidence_thresh_default = 0.70
        self.confidence_thresh_eval = 0.05
        self.MEAN = np.array([[[102.9801, 115.9465, 122.7717]]], dtype=np.float32)

        net = vgg16()
        class_num = len(self.classes)

        # load model
        net.create_architecture(class_num, tag='default')
        net.load_state_dict(t.load(weights, map_location=lambda storage, loc: storage))

        net.eval()
        if not t.cuda.is_available():
            net._device = 'cpu'
        net.to(net._device)

        self.net = net
        self.class_num = class_num
        self.net.zero_grad()

    def predict_all(self, blobs):
        img_h, img_w, im_scale = blobs['im_info']
        _, scores, bbox_pred, rois, rpn_cls_prob, rpn_bbox_pred, rpn_bbox_pred_rec = self.net.test_image_adv(
            blobs['data'], blobs['im_info'])

        rpn_bbox_pred_rec = rpn_bbox_pred_rec / im_scale
        boxes = rois[:, 1:5] / im_scale
        scores = np.reshape(scores, [scores.shape[0], -1])
        bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(t.from_numpy(boxes), t.from_numpy(box_deltas))
        pred_boxes = clip_boxes(pred_boxes, [img_h, img_w]).numpy()

        # Recover rpn bboxes
        rpn_bbox_pred = rpn_bbox_pred.view([-1, 4])
        rpn_cls_prob = t.stack([rpn_cls_prob[:, :, :, 0:self.net._num_anchors],
                                    rpn_cls_prob[:, :, :, self.net._num_anchors:]], dim=-1)
        rpn_cls_prob = rpn_cls_prob.contiguous().view((-1, 2))
        return scores, pred_boxes, rpn_cls_prob, rpn_bbox_pred_rec, rpn_bbox_pred

    def predict(self, blobs):
        """
        Test image
        :param im: bgr
        :return:
        """
        # Test
        img_h, img_w, im_scale = blobs['im_info']
        _, scores, bbox_pred, rois = self.net.test_image(blobs['data'], blobs['im_info'])

        boxes = rois[:, 1:5] / im_scale
        scores = np.reshape(scores, [scores.shape[0], -1])
        bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(t.from_numpy(boxes), t.from_numpy(box_deltas))
        pred_boxes = clip_boxes(pred_boxes, [img_h, img_w]).numpy()

        return scores, pred_boxes

    def postproc_dets(self, scores, boxes):
        thresh = 0.05
        max_per_image = 100
        # Visualize detections for each class
        dets_all_cls = [[] for _ in range(self.class_num)]
        # skip j = 0, because it's the background class
        for j in range(1, self.class_num):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            # TODO: Chow
            keep = nms(
                t.from_numpy(cls_boxes), t.from_numpy(cls_scores),
                cfg.TEST.NMS).numpy() if cls_dets.size > 0 else []
            # keep = nms(
            #     t.from_numpy(cls_dets),
            #     cfg.TEST.NMS).numpy() if cls_dets.size > 0 else []
            cls_dets = cls_dets[keep, :]
            dets_all_cls[j] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([dets_all_cls[j][:, -1]
                                      for j in range(1, self.class_num)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, self.class_num):
                    keep = np.where(dets_all_cls[j][:, -1] >= image_thresh)[0]
                    dets_all_cls[j] = dets_all_cls[j][keep, :]
        return dets_all_cls

    def convert_dets(self, dets_all_cls, thresh=0.5):
        dets_label = []
        dets_conf = []
        dets_box = []
        for label in range(len(dets_all_cls)):
            dets = dets_all_cls[label]
            if len(dets) == 0:
                continue
            inds = np.where(dets[:, -1] > thresh)[0]
            if len(inds) == 0:
                continue
            dets = dets[inds, :]
            dets_label.append([label] * dets.shape[0])
            dets_box.append(dets[:, :4])
            dets_conf.append(dets[:, -1])
        if len(dets_label):
            dets_conf = np.concatenate(dets_conf)
            dets_label = np.concatenate(dets_label)
            dets_box = np.concatenate(dets_box, 0)
        return np.array(dets_box), np.array(dets_label), np.array(dets_conf)

    def detect(self, x, iou_threshold=0.30, conf_threshold=0.70):
        x_ = x.copy()[:, :, :, [2, 1, 0]]
        scores, boxes = self.predict({'data': x_ * 255. - self.MEAN, 'im_info': (*x_.shape[1:3], 1.)})
        dets_all_cls = self.postproc_dets(scores, boxes)
        bboxes, labels, confs = self.convert_dets(dets_all_cls, thresh=conf_threshold)
        detections_query = []
        for oid in range(len(labels)):
            bbox = np.int32(np.round(bboxes[oid]))
            label = labels[oid]
            conf = confs[oid]
            detections_query.append((label, conf, *bbox))
        return np.asarray(detections_query)


class FRCNN(nn.Module):
    def __init__(self, model_img_size=(600, 600)):
        super(FRCNN, self).__init__()

        self.model_img_size = model_img_size
        self.confidence_thresh_default = 0.70
        self.confidence_thresh_eval = 0.05

        self.faster_rcnn = FRCNN_VGG16()
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets.
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = self.faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = self.faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()

    def forward(self, imgs, bboxes, labels, scale):
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois,
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi, at.tonumpy(bbox), at.tonumpy(label), self.loc_normalize_mean, self.loc_normalize_std)

        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(features, sample_roi, sample_roi_index)

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(at.tonumpy(bbox), anchor, img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label.data, self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(roi_loc.contiguous(), gt_roi_loc, gt_roi_label.data, self.roi_sigma)
        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        object_untargeted_loss = -(rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss)
        object_vanishing_loss = F.cross_entropy(rpn_score, t.zeros_like(gt_rpn_label)) + \
                                nn.CrossEntropyLoss()(roi_score, t.zeros_like(gt_roi_label))
        object_fabrication_loss = F.cross_entropy(rpn_score, t.ones_like(gt_rpn_label)) - \
                                  nn.CrossEntropyLoss()(roi_score, t.argmax(roi_score, dim=1))
        object_mislabeling_loss = -object_untargeted_loss
        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]
        losses = losses + [object_untargeted_loss,
                           object_vanishing_loss, object_fabrication_loss, object_mislabeling_loss]
        return LossTuple(*losses)

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def detect(self, x, iou_threshold=0.30, conf_threshold=0.70):
        x_local = x.copy() * 255

        # preprocess: (i) CxHxW and (ii) convert to t tensor
        x_tensor = t.from_numpy(x_local[0].transpose((2, 0, 1)))[None]

        _bboxes, _labels, _scores, _logits = self.faster_rcnn.predict(x_tensor, conf_threshold, iou_threshold)
        _bboxes = _bboxes[0][:, [1, 0, 3, 2]]  # ymin, xmin, ymax, xmax --> xmin, ymin, xmax, ymax
        _labels = np.expand_dims(_labels[0], axis=1)
        _scores = np.expand_dims(_scores[0], axis=1)
        _logits = _logits[0]
        detections_query = np.concatenate([_labels, _scores, _logits, _bboxes], axis=-1)
        return detections_query

    def compute_object_untargeted_gradient(self, x, detections):
        x_local = x.copy() * 255

        x_tensor = t.from_numpy(preprocess(x_local[0].transpose((2, 0, 1))))[None].cuda().float()
        x_tensor.requires_grad = True

        if detections is not None and len(detections) > 0:
            _bboxes = t.from_numpy(detections[np.newaxis, :, [-3, -4, -1, -2]]).float()
            _labels = t.from_numpy(detections[np.newaxis, :, 0]).int()
        else:
            _bboxes = t.from_numpy(np.zeros((1, 1, 4))).float()
            _labels = t.from_numpy(np.zeros((1, 1))).int()
        _scale = at.scalar(np.asarray([1.]))

        losses = self.forward(x_tensor, _bboxes, _labels, _scale)
        self.optimizer.zero_grad()
        self.faster_rcnn.zero_grad()
        if len(detections) > 0:
            losses.object_untargeted_loss.backward()
        else:
            losses.object_fabrication_loss.backward()
        return x_tensor.grad.data.cpu().numpy().transpose((0, 2, 3, 1))

    def compute_object_vanishing_gradient(self, x, detections=None):
        x_local = x.copy() * 255

        x_tensor = t.from_numpy(preprocess(x_local[0].transpose((2, 0, 1))))[None].cuda().float()
        x_tensor.requires_grad = True

        _bboxes = t.from_numpy(np.zeros((1, 1, 4))).float()
        _labels = t.from_numpy(np.zeros((1, 1))).int()
        _scale = at.scalar(np.asarray([1.]))

        losses = self.forward(x_tensor, _bboxes, _labels, _scale)
        self.optimizer.zero_grad()
        self.faster_rcnn.zero_grad()
        losses.object_vanishing_loss.backward()

        return x_tensor.grad.data.cpu().numpy().transpose((0, 2, 3, 1))

    def compute_object_fabrication_gradient(self, x, detections=None):
        x_local = x.copy() * 255

        x_tensor = t.from_numpy(preprocess(x_local[0].transpose((2, 0, 1))))[None].cuda().float()
        x_tensor.requires_grad = True

        _bboxes = t.from_numpy(np.zeros((1, 1, 4))).float()
        _labels = t.from_numpy(np.zeros((1, 1))).int()
        _scale = at.scalar(np.asarray([1.]))

        losses = self.forward(x_tensor, _bboxes, _labels, _scale)
        self.optimizer.zero_grad()
        self.faster_rcnn.zero_grad()
        losses.object_fabrication_loss.backward()

        return x_tensor.grad.data.cpu().numpy().transpose((0, 2, 3, 1))

    def compute_object_mislabeling_gradient(self, x, detections):
        x_local = x.copy() * 255

        x_tensor = t.from_numpy(preprocess(x_local[0].transpose((2, 0, 1))))[None].cuda().float()
        x_tensor.requires_grad = True

        _bboxes = t.from_numpy(detections[np.newaxis, :, [-3, -4, -1, -2]]).float()
        _labels = t.from_numpy(detections[np.newaxis, :, 0]).int()
        _scale = at.scalar(np.asarray([1.]))

        losses = self.forward(x_tensor, _bboxes, _labels, _scale)
        self.optimizer.zero_grad()
        self.faster_rcnn.zero_grad()
        losses.object_mislabeling_loss.backward()
        return x_tensor.grad.data.cpu().numpy().transpose((0, 2, 3, 1))

    @property
    def classes(self):
        return self.faster_rcnn.classes


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE: unlike origin implementation, we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negative and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float())  # ignore gt_label==-1 for rpn_loss
    return loc_loss
