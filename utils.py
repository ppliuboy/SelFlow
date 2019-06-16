import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from warp import tf_warp

def mvn(img):
    # minus mean color and divided by standard variance
    mean, var = tf.nn.moments(img, axes=[0, 1], keep_dims=True)
    img = (img - mean) / tf.sqrt(var + 1e-12)
    return img

def lrelu(x, leak=0.2, name='leaky_relu'):
    return tf.maximum(x, leak*x)

def imshow(img, re_normalize=False):
    if re_normalize:
        min_value = np.min(img)
        max_value = np.max(img)
        img = (img - min_value) / (max_value - min_value)
        img = img * 255
    elif np.max(img) <= 1.:
        img = img * 255
    img = img.astype('uint8')
    shape = img.shape
    if len(shape) == 2:
        img = np.repeat(np.expand_dims(img, -1), 3, -1)
    elif shape[2] == 1:
        img = np.repeat(img, 3, -1)
    plt.imshow(img)
    plt.show()
    
def rgb_bgr(img):
    tmp = np.copy(img[:, :, 0])
    img[:, :, 0] = np.copy(img[:, :, 2])
    img[:, :, 2] = np.copy(tmp)  
    return img
  
def compute_Fl(flow_gt, flow_est, mask):
    # F1 measure
    err = tf.multiply(flow_gt - flow_est, mask)
    err_norm = tf.norm(err, axis=-1)
    
    flow_gt_norm = tf.maximum(tf.norm(flow_gt, axis=-1), 1e-12)
    F1_logic = tf.logical_and(err_norm > 3, tf.divide(err_norm, flow_gt_norm) > 0.05)
    F1_logic = tf.cast(tf.logical_and(tf.expand_dims(F1_logic, -1), mask > 0), tf.float32)
    F1 = tf.reduce_sum(F1_logic) / (tf.reduce_sum(mask) + 1e-6)
    return F1    

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g is not None:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
        if grads != []:
            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads


def length_sq(x):
    return tf.reduce_sum(tf.square(x), 3, keepdims=True)    

def occlusion(flow_fw, flow_bw):
    x_shape = tf.shape(flow_fw)
    H = x_shape[1]
    W = x_shape[2]    
    flow_bw_warped = tf_warp(flow_bw, flow_fw, H, W)
    flow_fw_warped = tf_warp(flow_fw, flow_bw, H, W)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)
    mag_sq_bw = length_sq(flow_bw) + length_sq(flow_fw_warped)
    occ_thresh_fw =  0.01 * mag_sq_fw + 0.5
    occ_thresh_bw =  0.01 * mag_sq_bw + 0.5
    occ_fw = tf.cast(length_sq(flow_diff_fw) > occ_thresh_fw, tf.float32)
    occ_bw = tf.cast(length_sq(flow_diff_bw) > occ_thresh_bw, tf.float32)

    return occ_fw, occ_bw