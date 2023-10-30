import os
from os import listdir
from os.path import isfile, join
import numpy as np
from glob import glob
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, Nadam

np.random.seed(123)

import tensorflow as tf
import tensorflow.keras.metrics as tfm


class AUC(tfm.AUC):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(AUC, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(AUC, self).update_state(y_true, y_pred, sample_weight)

class MeanIoU(tfm.MeanIoU):
    def __init__(self, from_logits=False, num_classes=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(MeanIoU, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
       

class BinaryAccuracy(tfm.BinaryAccuracy):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(BinaryAccuracy, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)

class TruePositives(tfm.TruePositives):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(TruePositives, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(TruePositives, self).update_state(y_true, y_pred, sample_weight)


class FalsePositives(tfm.FalsePositives):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(FalsePositives, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(FalsePositives, self).update_state(y_true, y_pred, sample_weight)


class TrueNegatives(tfm.TrueNegatives):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(TrueNegatives, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(TrueNegatives, self).update_state(y_true, y_pred, sample_weight)


class FalseNegatives(tfm.FalseNegatives):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(FalseNegatives, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(FalseNegatives, self).update_state(y_true, y_pred, sample_weight)


class Precision(tfm.Precision):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(Precision, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)


class Recall(tfm.Recall):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(Recall, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)

class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, thres=0.5, name='Mean_IoU', dtype=None):
        super(MeanIoU, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.thres = thres
        self.total_cm = self.add_weight('total_confusion_matrix',
                                        shape=(num_classes, num_classes),
                                        initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])
        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])
        
        y_pred = tf.where(y_pred > self.thres, 1.0, 0.0)
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = tf.reshape(sample_weight, [-1])
        current_cm = tf.math.confusion_matrix(y_true,
                                              y_pred,
                                              self.num_classes,
                                              weights=sample_weight,
                                              dtype=self._dtype)
        
        return self.total_cm.assign_add(current_cm)

    def result(self):
        sum_over_row = tf.cast(tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = tf.cast(tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(tf.linalg.tensor_diag_part(self.total_cm), dtype=self._dtype)
        denominator = sum_over_row + sum_over_col - true_positives
        num_valid_entries = tf.reduce_sum(tf.cast(tf.math.not_equal(denominator, 0), dtype=self._dtype))
        iou = tf.math.divide_no_nan(true_positives, denominator)
        return tf.math.divide_no_nan(tf.reduce_sum(iou, name='Mean_IoU'), num_valid_entries)
    
    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        tf.keras.backend.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))
        
    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(MeanIoU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def dice_coeff(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    #Flatten
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_loss(y_true, y_pred):
    return 1-dice_coeff(y_true, y_pred)

def dice_loss2(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return 1 - score


# Define Dice loss function
def dice_loss3(y_true, y_pred, smooth=1e-7):
    y_pred = tf.cast(y_pred > 0.5, dtype=tf.float32)
    y_true = tf.cast(y_true > 0.5, dtype=tf.float32)
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    intersection = tf.reduce_sum(y_pred * y_true)
    loss = (2. * intersection + smooth) / (tf.reduce_sum(y_pred) + tf.reduce_sum(y_true) + smooth)
    return 1 - loss



def precision2(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall2(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def focal_loss(y_true, y_pred):
    """
    Focal loss
    :param y_true:
    :param y_pred:
    :return:
    """
    gamma = 2.
    alpha = 4.
    epsilon = 1.e-9

    y_true_c = tf.convert_to_tensor(y_true, tf.float32)
    y_pred_c = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred_c, epsilon)
    ce = tf.multiply(y_true_c, -tf.math.log(model_out))
    weight = tf.multiply(y_true_c, tf.pow(
        tf.subtract(1., model_out), gamma)
                         )
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=-1)
    return tf.reduce_mean(reduced_fl)


def ssim_loss(y_true, y_pred):
    """
    SSIM loss
    :param y_true:
    :param y_pred:
    :return:
    """
    return 1 - tf.image.ssim(y_true, y_pred, max_val=1)


def unet3p_hybrid_loss(y_true, y_pred):
    """
    Hybrid loss proposed in UNET 3+ (https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
    Hybrid loss for segmentation in three-level hierarchy – pixel, patch and map-level,
    which is able to capture both large-scale and fine structures with clear boundaries.
    :param y_true:
    :param y_pred:
    :return:
    """
    f_loss = focal_loss(y_true, y_pred)
    ms_ssim_loss = ssim_loss(y_true, y_pred)
    jacard_loss = iou_loss(y_true, y_pred)

    return f_loss + ms_ssim_loss + jacard_loss

def iou(y_true, y_pred):
    """
    Calculate intersection over union (IoU) between images
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true_f = K.flatten(K.clip(y_true, 0, 1))
    y_pred_f = K.flatten(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection / union


def iou_loss(y_true, y_pred):
    """
    Jaccard / IoU loss
    :param y_true:
    :param y_pred:
    :return:
    """
    return 1 - iou(y_true, y_pred)


def recall3(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall

def precision3(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    prec = (tp + smooth) / (tp + fp + smooth)
    return prec

