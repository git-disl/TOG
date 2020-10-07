from yolov3_utils.model import preprocess_true_boxes, yolo_boxes_and_scores, yolo_head, box_iou
from yolov3_utils.model import yolo_darknet53, yolo_mobilenetv1
from keras.layers import Input
from keras import backend as K
import tensorflow as tf
import numpy as np


class YOLOv3(object):
    anchors = np.asarray(
        [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])

    def __init__(self, weights, backbone, model_img_size, confidence_thresh_default, confidence_thresh_eval):
        self.model_img_size = model_img_size
        self.confidence_thresh_default = confidence_thresh_default
        self.confidence_thresh_eval = confidence_thresh_eval
        self.num_classes = len(self.classes)
        self.num_anchors = len(self.anchors)

        self.model = backbone(Input(shape=(None, None, 3)), self.num_anchors // 3, self.num_classes)
        self.model.load_weights(weights)

        self.encoded_detections = [tf.placeholder(dtype=tf.float32,
                                                  shape=self.model.output[layer].shape) for layer in range(3)]
        self.confidence_thresh = tf.placeholder(dtype=tf.float32, shape=())
        self.iou_thresh = tf.placeholder(dtype=tf.float32, shape=())
        self.nms_max_output_size = tf.placeholder(dtype=tf.int32, shape=())
        self.decoded_detections = self.build_decoding_graph()
        self.encoded_labels = [tf.placeholder(dtype=tf.float32,
                                              shape=self.model.output[layer].shape) for layer in range(3)]

        # Untargeted Attacks
        self.object_untargeted_loss = self.build_object_untargeted_loss()
        self.object_untargeted_gradient = tf.gradients(ys=self.object_untargeted_loss, xs=self.model.input)[0]

        # Targeted Attacks
        self.object_vanishing_loss = self.build_object_vanishing_loss()
        self.object_vanishing_gradient = tf.gradients(ys=self.object_vanishing_loss, xs=self.model.input)[0]

        self.object_fabrication_loss = self.build_object_fabrication_loss()
        self.object_fabrication_gradient = tf.gradients(ys=self.object_fabrication_loss, xs=self.model.input)[0]

        self.object_mislabeling_loss = self.build_object_mislabeling_loss()
        self.object_mislabeling_gradient = tf.gradients(ys=self.object_mislabeling_loss, xs=self.model.input)[0]

    def build_decoding_graph(self):
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        boxes, box_scores, box_presigmoid_probs = [], [], []
        for layer in range(len(anchor_mask)):
            _boxes, _box_scores, _box_presigmoid_probs = yolo_boxes_and_scores(
                self.encoded_detections[layer], self.anchors[anchor_mask[layer]],
                self.num_classes, self.model_img_size, self.model_img_size)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
            box_presigmoid_probs.append(_box_presigmoid_probs)
        boxes = K.concatenate(boxes, axis=0)
        box_scores = K.concatenate(box_scores, axis=0)
        box_presigmoid_probs = K.concatenate(box_presigmoid_probs, axis=0)

        mask = box_scores >= self.confidence_thresh
        class_ids, confidences, box_confidences, box_coordinates = [], [], [], []
        for c in range(self.num_classes):
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, self.nms_max_output_size,
                                                     iou_threshold=self.iou_thresh)
            class_boxes = K.gather(class_boxes, nms_index)
            class_box_scores = K.gather(class_box_scores, nms_index)
            class_id = K.ones_like(class_box_scores) * c
            box_confidence = K.gather(tf.boolean_mask(box_presigmoid_probs, mask[:, c]), nms_index)

            class_ids.append(class_id)
            confidences.append(class_box_scores)
            box_confidences.append(box_confidence)
            box_coordinates.append(class_boxes)

        class_ids = K.expand_dims(K.concatenate(class_ids, axis=0), axis=-1)
        confidences = K.expand_dims(K.concatenate(confidences, axis=0), axis=-1)
        box_confidences = K.concatenate(box_confidences, axis=0)
        box_coordinates = K.concatenate(box_coordinates, axis=0)

        ymin, xmin = tf.expand_dims(box_coordinates[..., 0], axis=-1), tf.expand_dims(box_coordinates[..., 1], axis=-1)
        ymax, xmax = tf.expand_dims(box_coordinates[..., 2], axis=-1), tf.expand_dims(box_coordinates[..., 3], axis=-1)
        box_coordinates = tf.concat(values=[xmin, ymin, xmax, ymax], axis=-1)

        return tf.concat([class_ids, confidences, box_confidences, box_coordinates], axis=-1)

    def detect(self, x, iou_threshold=0.45, conf_threshold=0.20, nms_max_output_size=400):
        # the model accepts [0-1] image, no preprocessing is needed
        encoded_detections = self.model.predict(x)
        decoded_detections = K.get_session().run(self.decoded_detections,
                                                 {self.encoded_detections[0]: encoded_detections[0],
                                                  self.encoded_detections[1]: encoded_detections[1],
                                                  self.encoded_detections[2]: encoded_detections[2],
                                                  self.iou_thresh: iou_threshold,
                                                  self.confidence_thresh: conf_threshold,
                                                  self.nms_max_output_size: nms_max_output_size})
        return decoded_detections

    def build_object_untargeted_loss(self):
        yolo_outputs = self.model.output
        y_true = self.encoded_labels
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
        grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(3)]
        loss = 0
        m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
        mf = K.cast(m, K.dtype(yolo_outputs[0]))

        for l in range(3):
            object_mask = y_true[l][..., 4:5]
            true_class_probs = y_true[l][..., 5:]

            grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l], self.anchors[anchor_mask[l]],
                                                         self.num_classes, input_shape, calc_loss=True)
            pred_box = K.concatenate([pred_xy, pred_wh])

            # Darknet raw box to calculate loss.
            raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
            raw_true_wh = K.log(y_true[l][..., 2:4] / self.anchors[anchor_mask[l]] * input_shape[::-1])
            raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
            box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

            # Find ignore mask, iterate over each of batch.
            ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
            object_mask_bool = K.cast(object_mask, 'bool')

            def loop_body(b, ignore_mask):
                true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
                iou = box_iou(pred_box[b], true_box)
                best_iou = K.max(iou, axis=-1)
                ignore_mask = ignore_mask.write(b, K.cast(best_iou < 0.45, K.dtype(true_box)))
                return b + 1, ignore_mask

            _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
            ignore_mask = ignore_mask.stack()
            ignore_mask = K.expand_dims(ignore_mask, -1)

            # K.binary_cross-entropy is helpful to avoid exp overflow.
            xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                           from_logits=True)
            wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
            confidence_loss = 0
            confidence_loss += object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True)
            confidence_loss += (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                         from_logits=True) * ignore_mask
            class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

            xy_loss = K.sum(xy_loss) / mf
            wh_loss = K.sum(wh_loss) / mf
            confidence_loss = K.sum(confidence_loss) / mf
            class_loss = K.sum(class_loss) / mf
            loss += xy_loss + wh_loss + confidence_loss + class_loss
        return -loss

    def compute_object_untargeted_gradient(self, x, detections):
        detections_ = np.asarray([detections[:, [-4, -3, -2, -1, 0]] if len(detections) > 0 else detections])
        encoded_labels = preprocess_true_boxes(detections_, input_shape=self.model_img_size,
                                               anchors=self.anchors, num_classes=self.num_classes)
        return K.get_session().run(self.object_untargeted_gradient,
                                   feed_dict={self.encoded_labels[0]: encoded_labels[0],
                                              self.encoded_labels[1]: encoded_labels[1],
                                              self.encoded_labels[2]: encoded_labels[2],
                                              self.model.input: x})

    def build_object_vanishing_loss(self):
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        input_shape = K.cast(K.shape(self.model.output[0])[1:3] * 32, K.dtype(self.encoded_labels[0]))
        confidence_loss = 0
        for layer in range(3):
            object_mask = self.encoded_labels[layer][..., 4:5]
            grid, raw_pred, pred_xy, pred_wh = yolo_head(self.model.output[layer], self.anchors[anchor_mask[layer]],
                                                         self.num_classes, input_shape, calc_loss=True)
            confidence_loss += K.sum(K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True))
        return confidence_loss

    def compute_object_vanishing_gradient(self, x, detections=None):
        detections_ = np.asarray([])
        encoded_labels = preprocess_true_boxes(detections_, input_shape=self.model_img_size,
                                               anchors=self.anchors, num_classes=self.num_classes)
        return K.get_session().run(self.object_vanishing_gradient, feed_dict={self.encoded_labels[0]: encoded_labels[0],
                                                                              self.encoded_labels[1]: encoded_labels[1],
                                                                              self.encoded_labels[2]: encoded_labels[2],
                                                                              self.model.input: x})

    def compute_object_vanishing_gradient_and_loss(self, x, detections=None):
        detections_ = np.asarray([])
        encoded_labels = preprocess_true_boxes(detections_, input_shape=self.model_img_size,
                                               anchors=self.anchors, num_classes=self.num_classes)
        return K.get_session().run([self.object_vanishing_gradient, self.object_vanishing_loss],
                                   feed_dict={self.encoded_labels[0]: encoded_labels[0],
                                              self.encoded_labels[1]: encoded_labels[1],
                                              self.encoded_labels[2]: encoded_labels[2],
                                              self.model.input: x})

    def build_object_fabrication_loss(self):
        return self.build_object_vanishing_loss()

    def compute_object_fabrication_gradient(self, x, detections=None):
        detections_ = np.asarray([])
        encoded_labels = preprocess_true_boxes(detections_, input_shape=self.model_img_size,
                                               anchors=self.anchors, num_classes=self.num_classes)
        encoded_labels[0][..., 4] = 1
        encoded_labels[1][..., 4] = 1
        encoded_labels[2][..., 4] = 1
        return K.get_session().run(self.object_fabrication_gradient,
                                   feed_dict={self.encoded_labels[0]: encoded_labels[0],
                                              self.encoded_labels[1]: encoded_labels[1],
                                              self.encoded_labels[2]: encoded_labels[2],
                                              self.model.input: x})

    def build_object_mislabeling_loss(self):
        return -self.build_object_untargeted_loss()

    def compute_object_mislabeling_gradient(self, x, detections):
        detections_ = np.asarray([detections[:, [-4, -3, -2, -1, 0]] if len(detections) > 0 else detections])
        encoded_labels = preprocess_true_boxes(detections_, input_shape=self.model_img_size,
                                               anchors=self.anchors, num_classes=self.num_classes)
        return K.get_session().run(self.object_mislabeling_gradient,
                                   feed_dict={self.encoded_labels[0]: encoded_labels[0],
                                              self.encoded_labels[1]: encoded_labels[1],
                                              self.encoded_labels[2]: encoded_labels[2],
                                              self.model.input: x})

    def compute_object_mislabeling_gradient_and_loss(self, x, detections):
        detections_ = np.asarray([detections[:, [-4, -3, -2, -1, 0]] if len(detections) > 0 else detections])
        encoded_labels = preprocess_true_boxes(detections_, input_shape=self.model_img_size,
                                               anchors=self.anchors, num_classes=self.num_classes)
        return K.get_session().run([self.object_mislabeling_gradient, self.object_mislabeling_loss],
                                   feed_dict={self.encoded_labels[0]: encoded_labels[0],
                                              self.encoded_labels[1]: encoded_labels[1],
                                              self.encoded_labels[2]: encoded_labels[2],
                                              self.model.input: x})


class YOLOv3_Darknet53(YOLOv3):
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self, weights, model_img_size=(416, 416), confidence_thresh_default=0.20, confidence_thresh_eval=0.01):
        super().__init__(weights, yolo_darknet53,
                         model_img_size, confidence_thresh_default, confidence_thresh_eval)


class YOLOv3_MobileNetV1(YOLOv3):
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self, weights, model_img_size=(416, 416), confidence_thresh_default=0.20, confidence_thresh_eval=0.01):
        super().__init__(weights, yolo_mobilenetv1,
                         model_img_size, confidence_thresh_default, confidence_thresh_eval)


# COCO Dataset
class YOLOv3_Darknet53_COCO(YOLOv3):
    classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'trafficlight',
               'firehydrant', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sportsball', 'kite', 'baseballbat', 'baseballglove', 'skateboard', 'surfboard',
               'tennisracket', 'bottle', 'wineglass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
               'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
               'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddybear', 'hairdrier', 'toothbrush']

    def __init__(self, weights, model_img_size=(416, 416), confidence_thresh_default=0.20, confidence_thresh_eval=0.01):
        print(len(self.classes))
        super().__init__(weights, yolo_darknet53,
                         model_img_size, confidence_thresh_default, confidence_thresh_eval)
