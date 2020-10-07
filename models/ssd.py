from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate
from ssd_utils.keras_layer_L2Normalization import L2Normalization
from ssd_utils.ssd_input_encoder import SSDInputEncoder
from ssd_utils.keras_layer_AnchorBoxes import AnchorBoxes
from keras.regularizers import l2
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import numpy as np


class SSD(object):
    classes = ['background',
               'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self, weights, model, input_encoder, model_img_size):
        self.n_classes = len(self.classes)
        self.model_img_height, self.model_img_width = model_img_size[0], model_img_size[1]
        self.model_img_size = model_img_size[:2]
        self.confidence_thresh_default = 0.50
        self.confidence_thresh_eval = 0.01

        self.model = model
        self.model.load_weights(weights, by_name=True)
        self.input_encoder = input_encoder

        predictions = self.model.get_layer('predictions').output
        self.encoded_detections = tf.placeholder(dtype=tf.float32, shape=predictions._keras_shape)
        self.confidence_thresh = tf.placeholder(dtype=tf.float32, shape=())
        self.iou_thresh = tf.placeholder(dtype=tf.float32, shape=())
        self.nms_max_output_size = tf.placeholder(dtype=tf.int32, shape=())
        self.top_k = tf.placeholder(dtype=tf.int32, shape=())
        self.decoded_detections = self.build_decoding_graph()
        self.encoded_labels = tf.placeholder(dtype=tf.float32, shape=predictions._keras_shape)

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
        # Convert anchor box offsets to image offsets.
        # cx = cx_pred * cx_variance * w_anchor + cx_anchor
        cx = self.encoded_detections[..., -12] * self.encoded_detections[..., -4] * self.encoded_detections[..., -6] + \
             self.encoded_detections[..., -8]
        # cy = cy_pred * cy_variance * h_anchor + cy_anchor
        cy = self.encoded_detections[..., -11] * self.encoded_detections[..., -3] * self.encoded_detections[..., -5] + \
             self.encoded_detections[..., -7]
        # w = exp(w_pred * variance_w) * w_anchor
        w = tf.exp(self.encoded_detections[..., -10] * self.encoded_detections[..., -2]) * self.encoded_detections[
            ..., -6]
        # h = exp(h_pred * variance_h) * h_anchor
        h = tf.exp(self.encoded_detections[..., -9] * self.encoded_detections[..., -1]) * self.encoded_detections[
            ..., -5]

        # Convert 'centroids' to 'corners'.
        xmin = tf.expand_dims((cx - 0.5 * w) * self.model_img_width, axis=-1)
        ymin = tf.expand_dims((cy - 0.5 * h) * self.model_img_height, axis=-1)
        xmax = tf.expand_dims((cx + 0.5 * w) * self.model_img_width, axis=-1)
        ymax = tf.expand_dims((cy + 0.5 * h) * self.model_img_height, axis=-1)

        # Concatenate the one-hot class confidences and the converted box coordinates to
        # form the decoded predictions tensor.
        y_pred = tf.concat(values=[self.encoded_detections[..., :-12], xmin, ymin, xmax, ymax], axis=-1)

        #############################################################################################
        # 2. Perform confidence thresholding, per-class non-maximum suppression, and top-k filtering.
        #############################################################################################

        # Create a function that filters the predictions for the given batch item. Specifically, it performs:
        # - confidence thresholding
        # - non-maximum suppression (NMS)
        # - top-k filtering
        def filter_predictions(batch_item):
            # Create a function that filters the predictions for one single class.
            def filter_single_class(index):
                # From a tensor of shape (n_boxes, n_classes + 4 coordinates) extract
                # a tensor of shape (n_boxes, 1 + 4 coordinates) that contains the
                # confidence values for just one class, determined by `index`.
                box_confidences_softmax = tf.nn.softmax(batch_item[..., :-4])
                confidences = tf.expand_dims(box_confidences_softmax[..., index], axis=-1)
                class_id = tf.fill(dims=tf.shape(confidences), value=tf.to_float(index))
                box_confidences = batch_item[..., :-4]
                box_coordinates = batch_item[..., -4:]

                single_class = tf.concat([class_id, confidences, box_confidences, box_coordinates], axis=-1)

                # Apply confidence thresholding with respect to the class defined by `index`.
                threshold_met = single_class[:, 1] > self.confidence_thresh
                single_class = tf.boolean_mask(tensor=single_class, mask=threshold_met)

                # If any boxes made the threshold, perform NMS.
                def perform_nms():
                    scores = single_class[..., 1]
                    # tf.image.non_max_suppression() needs the box coordinates in the format (ymin, xmin, ymax, xmax).
                    xmin = tf.expand_dims(single_class[..., -4], axis=-1)
                    ymin = tf.expand_dims(single_class[..., -3], axis=-1)
                    xmax = tf.expand_dims(single_class[..., -2], axis=-1)
                    ymax = tf.expand_dims(single_class[..., -1], axis=-1)
                    boxes = tf.concat(values=[ymin, xmin, ymax, xmax], axis=-1)
                    maxima_indices = tf.image.non_max_suppression(boxes=boxes, scores=scores,
                                                                  max_output_size=self.nms_max_output_size,
                                                                  iou_threshold=self.iou_thresh,
                                                                  name='non_maximum_suppression')
                    maxima = tf.gather(params=single_class, indices=maxima_indices, axis=0)
                    return maxima

                def no_confident_predictions():
                    return tf.constant(value=0.0, shape=(1, 27))

                single_class_nms = tf.cond(tf.equal(tf.size(single_class), 0), no_confident_predictions, perform_nms)

                # Make sure `single_class` is exactly `self.nms_max_output_size` elements long.
                padded_single_class = tf.pad(tensor=single_class_nms,
                                             paddings=[[0, self.nms_max_output_size - tf.shape(single_class_nms)[0]],
                                                       [0, 0]],
                                             mode='CONSTANT',
                                             constant_values=0.0)

                return padded_single_class

            # Iterate `filter_single_class()` over all class indices.
            filtered_single_classes = tf.map_fn(fn=lambda i: filter_single_class(i),
                                                elems=tf.range(1, self.n_classes),
                                                dtype=tf.float32,
                                                parallel_iterations=128,
                                                back_prop=False,
                                                swap_memory=False,
                                                infer_shape=True,
                                                name='loop_over_classes')

            # Concatenate the filtered results for all individual classes to one tensor.
            filtered_predictions = tf.reshape(tensor=filtered_single_classes, shape=(-1, 27))

            def top_k_prediction():
                return tf.gather(params=filtered_predictions,
                                 indices=tf.nn.top_k(filtered_predictions[:, 1], k=self.top_k, sorted=True).indices,
                                 axis=0)

            def pad_and_top_k_prediction():
                padded_predictions = tf.pad(tensor=filtered_predictions,
                                            paddings=[[0, self.top_k - tf.shape(filtered_predictions)[0]], [0, 0]],
                                            mode='CONSTANT',
                                            constant_values=0.0)
                return tf.gather(params=padded_predictions,
                                 indices=tf.nn.top_k(padded_predictions[:, 1], k=self.top_k, sorted=True).indices,
                                 axis=0)

            top_k_boxes = tf.cond(tf.greater_equal(tf.shape(filtered_predictions)[0], self.top_k),
                                  top_k_prediction, pad_and_top_k_prediction)

            return top_k_boxes

        # Iterate `filter_predictions()` over all batch items.
        return tf.map_fn(fn=lambda x: filter_predictions(x), elems=y_pred, dtype=None,
                         parallel_iterations=128, back_prop=False, swap_memory=False,
                         infer_shape=True, name='loop_over_batch')

    def detect(self, x, iou_threshold=0.45, conf_threshold=0.50, top_k=400, nms_max_output_size=400):
        # the model accepts [0-255] image only
        encoded_detections = self.model.predict(x.copy() * 255)
        decoded_detections = K.get_session().run(self.decoded_detections,
                                                 {self.encoded_detections: encoded_detections,
                                                  self.iou_thresh: iou_threshold,
                                                  self.confidence_thresh: conf_threshold,
                                                  self.top_k: top_k,
                                                  self.nms_max_output_size: nms_max_output_size})
        detected_objects = []
        for row in decoded_detections[0]:
            class_id = int(row[0])
            if class_id == 0:
                continue
            detected_objects.append(row)
        return np.asarray(detected_objects)

    def build_object_untargeted_loss(self):
        classification_loss = tf.to_float(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.encoded_labels[:, :, :-12],
                                                    logits=self.model.output[:, :, :-12]))
        localization_loss = tf.to_float(self.smooth_L1_loss(self.encoded_labels[:, :, -12:-8],
                                                            self.model.output[:, :, -12:-8]))

        positives = tf.to_float(tf.reduce_max(self.encoded_labels[:, :, 1:-12], axis=-1))
        class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)
        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)
        return -tf.reduce_mean(class_loss + loc_loss)

    def compute_object_untargeted_gradient(self, x, detections):
        detections_ = detections[:, [0, -4, -3, -2, -1]] if len(detections) > 0 else detections
        encoded_labels = self.input_encoder([detections_])
        return K.get_session().run(self.object_untargeted_gradient, feed_dict={self.encoded_labels: encoded_labels,
                                                                               self.model.input: x.copy() * 255})

    def build_object_vanishing_loss(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.model.output[:, :, :-12], labels=self.encoded_labels[:, :, :-12]))

    def compute_object_vanishing_gradient(self, x, detections=None):
        encoded_labels = self.input_encoder([np.asarray([])])
        return K.get_session().run(self.object_vanishing_gradient, feed_dict={self.encoded_labels: encoded_labels,
                                                                              self.model.input: x.copy() * 255})

    def build_object_fabrication_loss(self):
        return -self.build_object_vanishing_loss()

    def compute_object_fabrication_gradient(self, x, detections=None):
        encoded_labels = self.input_encoder([np.asarray([])])
        return K.get_session().run(self.object_fabrication_gradient, feed_dict={self.encoded_labels: encoded_labels,
                                                                                self.model.input: x.copy() * 255})

    def build_object_mislabeling_loss(self):
        return -self.build_object_untargeted_loss()

    def compute_object_mislabeling_gradient(self, x, detections):
        detections_ = detections[:, [0, -4, -3, -2, -1]] if len(detections) > 0 else detections
        encoded_labels = self.input_encoder([detections_])
        return K.get_session().run(self.object_mislabeling_gradient, feed_dict={self.encoded_labels: encoded_labels,
                                                                                self.model.input: x.copy() * 255})

    @staticmethod
    def smooth_L1_loss(y_true, y_pred):
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)


class SSD300(SSD):
    def __init__(self, weights, model_img_size=(300, 300, 3), l2_reg=0.0005,
                 scales=(0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05), aspect_ratios=([1.0, 2.0, 0.5],
                                                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                                                 [1.0, 2.0, 0.5],
                                                                                 [1.0, 2.0, 0.5]),
                 two_boxes_for_ar1=True, steps=(8, 16, 32, 64, 100, 300), offsets=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                 clip_boxes=False, variances=(0.1, 0.1, 0.2, 0.2), coords='centroids', normalize_coords=True,
                 subtract_mean=(123, 117, 104), divide_by_stddev=None, swap_channels=(2, 1, 0)):

        if len(model_img_size) == 2:
            model_img_size = (*model_img_size, 3)
        n_classes = len(SSD.classes)
        model_img_height, model_img_width = model_img_size[0], model_img_size[1]

        ############################################################################
        # Compute the anchor box parameters.
        ############################################################################

        # Compute the number of boxes to be predicted per cell for each predictor layer.
        # We need this so that we know how many channels the predictor layers need to have.
        n_boxes = []
        for ar in aspect_ratios:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))

        ############################################################################
        # Define functions for the Lambda layers below.
        ############################################################################

        def identity_layer(tensor):
            return tensor

        def input_mean_normalization(tensor):
            return tensor - np.array(subtract_mean)

        def input_stddev_normalization(tensor):
            return tensor / np.array(divide_by_stddev)

        def input_channel_swap(tensor):
            return K.stack(
                [tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]]], axis=-1)

        ############################################################################
        # Build the network.
        ############################################################################
        x = Input(shape=model_img_size)

        # The following identity layer is only needed so that the subsequent lambda layers can be optional.
        x1 = Lambda(identity_layer, output_shape=model_img_size, name='identity_layer')(x)
        if not (subtract_mean is None):
            x1 = Lambda(input_mean_normalization, output_shape=model_img_size, name='input_mean_normalization')(x1)
        if not (divide_by_stddev is None):
            x1 = Lambda(input_stddev_normalization, output_shape=model_img_size, name='input_stddev_normalization')(x1)
        if swap_channels:
            x1 = Lambda(input_channel_swap, output_shape=model_img_size, name='input_channel_swap')(x1)

        conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv1_1')(x1)
        conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv1_2')(conv1_1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

        conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv2_1')(pool1)
        conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv2_2')(conv2_1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

        conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv3_1')(pool2)
        conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv3_2')(conv3_1)
        conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv3_3')(conv3_2)
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

        conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv4_1')(pool3)
        conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv4_2')(conv4_1)
        conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv4_3')(conv4_2)
        pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

        conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv5_1')(pool4)
        conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv5_2')(conv5_1)
        conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv5_3')(conv5_2)
        pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

        fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc6')(pool5)

        fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='fc7')(fc6)

        conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv6_1')(fc7)
        conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
        conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid',
                         kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2')(conv6_1)

        conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv7_1')(conv6_2)
        conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
        conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid',
                         kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2')(conv7_1)

        conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv8_1')(conv7_2)
        conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid',
                         kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2')(conv8_1)

        conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv9_1')(conv8_2)
        conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid',
                         kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2')(conv9_1)

        # Feed conv4_3 into the L2 normalization layer
        conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3)

        # <--- Build the convolutional predictor layers on top of the base network --->

        # We predict `n_classes` confidence values for each box, hence the confidence predictors have depth
        # `n_boxes * n_classes`
        # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
        conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same',
                                        kernel_initializer='he_normal',
                                        kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_conf')(conv4_3_norm)
        fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='fc7_mbox_conf')(fc7)
        conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_conf')(conv6_2)
        conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_conf')(conv7_2)
        conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf')(conv8_2)
        conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf')(conv9_2)
        # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
        # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
        conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                       kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_loc')(conv4_3_norm)
        fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='fc7_mbox_loc')(fc7)
        conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                  kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_loc')(conv6_2)
        conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                  kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_loc')(conv7_2)
        conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                  kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_loc')(conv8_2)
        conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                  kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_loc')(conv9_2)

        # <--- Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation,
        #      so I'll keep their layer names) --->

        # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
        conv4_3_norm_mbox_priorbox = AnchorBoxes(model_img_height, model_img_width, this_scale=scales[0],
                                                 next_scale=scales[1],
                                                 aspect_ratios=aspect_ratios[0],
                                                 two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0],
                                                 this_offsets=offsets[0], clip_boxes=clip_boxes,
                                                 variances=variances, coords=coords, normalize_coords=normalize_coords,
                                                 name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)
        fc7_mbox_priorbox = AnchorBoxes(model_img_height, model_img_width, this_scale=scales[1],
                                        next_scale=scales[2],
                                        aspect_ratios=aspect_ratios[1],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1],
                                        this_offsets=offsets[1], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='fc7_mbox_priorbox')(fc7_mbox_loc)
        conv6_2_mbox_priorbox = AnchorBoxes(model_img_height, model_img_width, this_scale=scales[2],
                                            next_scale=scales[3],
                                            aspect_ratios=aspect_ratios[2],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2],
                                            this_offsets=offsets[2], clip_boxes=clip_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords,
                                            name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
        conv7_2_mbox_priorbox = AnchorBoxes(model_img_height, model_img_width, this_scale=scales[3],
                                            next_scale=scales[4],
                                            aspect_ratios=aspect_ratios[3],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3],
                                            this_offsets=offsets[3], clip_boxes=clip_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords,
                                            name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
        conv8_2_mbox_priorbox = AnchorBoxes(model_img_height, model_img_width, this_scale=scales[4],
                                            next_scale=scales[5],
                                            aspect_ratios=aspect_ratios[4],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4],
                                            this_offsets=offsets[4], clip_boxes=clip_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords,
                                            name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
        conv9_2_mbox_priorbox = AnchorBoxes(model_img_height, model_img_width, this_scale=scales[5],
                                            next_scale=scales[6],
                                            aspect_ratios=aspect_ratios[5],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5],
                                            this_offsets=offsets[5], clip_boxes=clip_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords,
                                            name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)

        # <--- Reshape --->
        # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
        # We want the classes isolated in the last axis to perform softmax on them
        conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='conv4_3_norm_mbox_conf_reshape')(
            conv4_3_norm_mbox_conf)
        fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
        conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
        conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
        conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
        conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
        # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
        # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
        conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
        fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
        conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
        conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
        conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
        conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)
        # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
        conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(
            conv4_3_norm_mbox_priorbox)
        fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
        conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
        conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
        conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
        conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

        # <--- Concatenate the predictions from the different layers --->

        # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
        # so we want to concatenate along axis 1, the number of boxes per layer
        # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
        mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                           fc7_mbox_conf_reshape,
                                                           conv6_2_mbox_conf_reshape,
                                                           conv7_2_mbox_conf_reshape,
                                                           conv8_2_mbox_conf_reshape,
                                                           conv9_2_mbox_conf_reshape])

        # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
        mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                         fc7_mbox_loc_reshape,
                                                         conv6_2_mbox_loc_reshape,
                                                         conv7_2_mbox_loc_reshape,
                                                         conv8_2_mbox_loc_reshape,
                                                         conv9_2_mbox_loc_reshape])

        # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
        mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                                   fc7_mbox_priorbox_reshape,
                                                                   conv6_2_mbox_priorbox_reshape,
                                                                   conv7_2_mbox_priorbox_reshape,
                                                                   conv8_2_mbox_priorbox_reshape,
                                                                   conv9_2_mbox_priorbox_reshape])

        # Concatenate the class and box predictions and the anchors to one large predictions vector
        # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
        predictions = Concatenate(axis=2, name='predictions')([mbox_conf, mbox_loc, mbox_priorbox])

        model = Model(inputs=x, outputs=predictions)
        input_encoder = SSDInputEncoder(img_height=model_img_height, img_width=model_img_width,
                                        n_classes=n_classes - 1,
                                        scales=scales, predictor_sizes=[conv4_3_norm_mbox_conf._keras_shape[1:3],
                                                                        fc7_mbox_conf._keras_shape[1:3],
                                                                        conv6_2_mbox_conf._keras_shape[1:3],
                                                                        conv7_2_mbox_conf._keras_shape[1:3],
                                                                        conv8_2_mbox_conf._keras_shape[1:3],
                                                                        conv9_2_mbox_conf._keras_shape[1:3]],
                                        aspect_ratios_per_layer=aspect_ratios, two_boxes_for_ar1=two_boxes_for_ar1,
                                        steps=steps, offsets=offsets, clip_boxes=clip_boxes, variances=variances,
                                        normalize_coords=normalize_coords)

        super().__init__(weights, model, input_encoder, model_img_size)


class SSD512(SSD):
    def __init__(self, weights, model_img_size=(512, 512, 3), l2_reg=0.0005,
                 scales=(0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05), aspect_ratios=([1.0, 2.0, 0.5],
                                                                                      [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                                                      [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                                                      [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                                                      [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                                                      [1.0, 2.0, 0.5],
                                                                                      [1.0, 2.0, 0.5]),
                 two_boxes_for_ar1=True, steps=(8, 16, 32, 64, 128, 256, 512),
                 offsets=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                 clip_boxes=False, variances=(0.1, 0.1, 0.2, 0.2), coords='centroids', normalize_coords=True,
                 subtract_mean=(123, 117, 104), divide_by_stddev=None, swap_channels=(2, 1, 0)):

        if len(model_img_size) == 2:
            model_img_size = (*model_img_size, 3)
        n_classes = len(SSD.classes)
        model_img_height, model_img_width = model_img_size[0], model_img_size[1]

        ############################################################################
        # Compute the anchor box parameters.
        ############################################################################

        # Compute the number of boxes to be predicted per cell for each predictor layer.
        # We need this so that we know how many channels the predictor layers need to have.
        n_boxes = []
        for ar in aspect_ratios:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))

        ############################################################################
        # Define functions for the Lambda layers below.
        ############################################################################

        def identity_layer(tensor):
            return tensor

        def input_mean_normalization(tensor):
            return tensor - np.array(subtract_mean)

        def input_stddev_normalization(tensor):
            return tensor / np.array(divide_by_stddev)

        def input_channel_swap(tensor):
            return K.stack(
                [tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]]], axis=-1)

        ############################################################################
        # Build the network.
        ############################################################################

        x = Input(shape=model_img_size)

        # The following identity layer is only needed so that the subsequent lambda layers can be optional.
        x1 = Lambda(identity_layer, output_shape=model_img_size, name='identity_layer')(x)
        if not (subtract_mean is None):
            x1 = Lambda(input_mean_normalization, output_shape=model_img_size, name='input_mean_normalization')(x1)
        if not (divide_by_stddev is None):
            x1 = Lambda(input_stddev_normalization, output_shape=model_img_size, name='input_stddev_normalization')(x1)
        if swap_channels:
            x1 = Lambda(input_channel_swap, output_shape=model_img_size, name='input_channel_swap')(x1)

        conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv1_1')(x1)
        conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv1_2')(conv1_1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

        conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv2_1')(pool1)
        conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv2_2')(conv2_1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

        conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv3_1')(pool2)
        conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv3_2')(conv3_1)
        conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv3_3')(conv3_2)
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

        conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv4_1')(pool3)
        conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv4_2')(conv4_1)
        conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv4_3')(conv4_2)
        pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

        conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv5_1')(pool4)
        conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv5_2')(conv5_1)
        conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv5_3')(conv5_2)
        pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

        fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc6')(pool5)

        fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='fc7')(fc6)

        conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv6_1')(fc7)
        conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
        conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid',
                         kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2')(conv6_1)

        conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv7_1')(conv6_2)
        conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
        conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid',
                         kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2')(conv7_1)

        conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv8_1')(conv7_2)
        conv8_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv8_padding')(conv8_1)
        conv8_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid',
                         kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2')(conv8_1)

        conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_reg), name='conv9_1')(conv8_2)
        conv9_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv9_padding')(conv9_1)
        conv9_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid',
                         kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2')(conv9_1)

        conv10_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(l2_reg), name='conv10_1')(conv9_2)
        conv10_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv10_padding')(conv10_1)
        conv10_2 = Conv2D(256, (4, 4), strides=(1, 1), activation='relu', padding='valid',
                          kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv10_2')(conv10_1)

        # Feed conv4_3 into the L2 normalization layer
        conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3)

        # <--- Build the convolutional predictor layers on top of the base network --->

        # We predict `n_classes` confidence values for each box, hence the confidence predictors have depth
        # `n_boxes * n_classes`
        # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
        conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same',
                                        kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                                        name='conv4_3_norm_mbox_conf')(conv4_3_norm)
        fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='fc7_mbox_conf')(fc7)
        conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_conf')(conv6_2)
        conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_conf')(conv7_2)
        conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf')(conv8_2)
        conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf')(conv9_2)
        conv10_2_mbox_conf = Conv2D(n_boxes[6] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_reg), name='conv10_2_mbox_conf')(conv10_2)
        # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
        # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
        conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                       kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_loc')(conv4_3_norm)
        fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='fc7_mbox_loc')(fc7)
        conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                  kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_loc')(conv6_2)
        conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                  kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_loc')(conv7_2)
        conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                  kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_loc')(conv8_2)
        conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                  kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_loc')(conv9_2)
        conv10_2_mbox_loc = Conv2D(n_boxes[6] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv10_2_mbox_loc')(conv10_2)

        # <--- Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation,
        #      so I'll keep their layer names) --->

        # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
        conv4_3_norm_mbox_priorbox = AnchorBoxes(model_img_height, model_img_width, this_scale=scales[0],
                                                 next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                                 two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0],
                                                 this_offsets=offsets[0], clip_boxes=clip_boxes,
                                                 variances=variances, coords=coords, normalize_coords=normalize_coords,
                                                 name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)
        fc7_mbox_priorbox = AnchorBoxes(model_img_height, model_img_width, this_scale=scales[1],
                                        next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1],
                                        this_offsets=offsets[1], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='fc7_mbox_priorbox')(fc7_mbox_loc)
        conv6_2_mbox_priorbox = AnchorBoxes(model_img_height, model_img_width, this_scale=scales[2],
                                            next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2],
                                            this_offsets=offsets[2], clip_boxes=clip_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords,
                                            name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
        conv7_2_mbox_priorbox = AnchorBoxes(model_img_height, model_img_width, this_scale=scales[3],
                                            next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3],
                                            this_offsets=offsets[3], clip_boxes=clip_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords,
                                            name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
        conv8_2_mbox_priorbox = AnchorBoxes(model_img_height, model_img_width, this_scale=scales[4],
                                            next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4],
                                            this_offsets=offsets[4], clip_boxes=clip_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords,
                                            name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
        conv9_2_mbox_priorbox = AnchorBoxes(model_img_height, model_img_width, this_scale=scales[5],
                                            next_scale=scales[6], aspect_ratios=aspect_ratios[5],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5],
                                            this_offsets=offsets[5], clip_boxes=clip_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords,
                                            name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)
        conv10_2_mbox_priorbox = AnchorBoxes(model_img_height, model_img_width, this_scale=scales[6],
                                             next_scale=scales[7], aspect_ratios=aspect_ratios[6],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[6],
                                             this_offsets=offsets[6], clip_boxes=clip_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords,
                                             name='conv10_2_mbox_priorbox')(conv10_2_mbox_loc)

        # <--- Reshape --->
        # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
        # We want the classes isolated in the last axis to perform softmax on them
        conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='conv4_3_norm_mbox_conf_reshape')(
            conv4_3_norm_mbox_conf)
        fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
        conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
        conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
        conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
        conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
        conv10_2_mbox_conf_reshape = Reshape((-1, n_classes),
                                             name='conv10_2_mbox_conf_reshape')(conv10_2_mbox_conf)
        # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
        # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
        conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
        fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
        conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
        conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
        conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
        conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)
        conv10_2_mbox_loc_reshape = Reshape((-1, 4), name='conv10_2_mbox_loc_reshape')(conv10_2_mbox_loc)
        # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
        conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(
            conv4_3_norm_mbox_priorbox)
        fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
        conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
        conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
        conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
        conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)
        conv10_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv10_2_mbox_priorbox_reshape')(conv10_2_mbox_priorbox)

        # <--- Concatenate the predictions from the different layers --->

        # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
        # so we want to concatenate along axis 1, the number of boxes per layer
        # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
        mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                           fc7_mbox_conf_reshape,
                                                           conv6_2_mbox_conf_reshape,
                                                           conv7_2_mbox_conf_reshape,
                                                           conv8_2_mbox_conf_reshape,
                                                           conv9_2_mbox_conf_reshape,
                                                           conv10_2_mbox_conf_reshape])

        # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
        mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                         fc7_mbox_loc_reshape,
                                                         conv6_2_mbox_loc_reshape,
                                                         conv7_2_mbox_loc_reshape,
                                                         conv8_2_mbox_loc_reshape,
                                                         conv9_2_mbox_loc_reshape,
                                                         conv10_2_mbox_loc_reshape])

        # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
        mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                                   fc7_mbox_priorbox_reshape,
                                                                   conv6_2_mbox_priorbox_reshape,
                                                                   conv7_2_mbox_priorbox_reshape,
                                                                   conv8_2_mbox_priorbox_reshape,
                                                                   conv9_2_mbox_priorbox_reshape,
                                                                   conv10_2_mbox_priorbox_reshape])

        # Concatenate the class and box predictions and the anchors to one large predictions vector
        # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
        predictions = Concatenate(axis=2, name='predictions')([mbox_conf, mbox_loc, mbox_priorbox])

        model = Model(inputs=x, outputs=predictions)
        input_encoder = SSDInputEncoder(img_height=model_img_height, img_width=model_img_width,
                                        n_classes=n_classes - 1,
                                        scales=scales, predictor_sizes=[conv4_3_norm_mbox_conf._keras_shape[1:3],
                                                                        fc7_mbox_conf._keras_shape[1:3],
                                                                        conv6_2_mbox_conf._keras_shape[1:3],
                                                                        conv7_2_mbox_conf._keras_shape[1:3],
                                                                        conv8_2_mbox_conf._keras_shape[1:3],
                                                                        conv9_2_mbox_conf._keras_shape[1:3],
                                                                        conv10_2_mbox_conf._keras_shape[1:3]],
                                        aspect_ratios_per_layer=aspect_ratios, two_boxes_for_ar1=two_boxes_for_ar1,
                                        steps=steps, offsets=offsets, clip_boxes=clip_boxes, variances=variances,
                                        normalize_coords=normalize_coords)

        super().__init__(weights, model, input_encoder, model_img_size)
