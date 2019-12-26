import tensorflow as tf
import numpy as np
import os
from config import config

def log_gaussian_prob_density(mean: "3d tensor", log_std: "scalar tensor", x: "3d tensor"):
    # gp = tf.exp(-1 * tf.square(x - mean) / (2 * var)) / ((2 * np.pi) ** 0.5 * std)
    log_gp = - 0.5 * tf.log(2 * np.pi) - log_std - 0.5 * ((x - mean) / tf.exp(log_std)) ** 2
    return tf.reduce_sum(log_gp, axis = 2)

def normalize(x: "2d tensor"):
    mean = tf.reduce_mean(x, axis = 0)
    std = (tf.reduce_sum((x - mean) ** 2, axis = 0) / (tf.to_float(tf.shape(x)[0]) - 1.0)) ** 0.5
    return (x - mean) / (std + 1e-12)

def zero_center(x: "2d tensor"):
    mean = tf.reduce_mean(x, axis = 0)
    return x - mean

def get_tf_session():
    if not config.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sess_config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config = sess_config)

    if config.sess_nan_test:
        from tensorflow.python import debug as tf_debug
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    return sess
