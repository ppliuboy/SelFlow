import tensorflow as tf
from data_augmentation import flow_resize

def mse_loss(diff, mask):
    diff_square = tf.square(diff)
    diff_square = tf.multiply(diff_square, mask)
    diff_square_sum = tf.reduce_sum(diff_square)
    loss_mean = diff_square_sum / (tf.reduce_sum(mask) * 2 + 1e-6)
    batch_size = tf.to_float(tf.shape(diff)[0])
    loss_sum = diff_square_sum / batch_size        
    return loss_mean, loss_sum
    
def abs_loss(diff, mask):
    diff_abs = tf.abs(diff)
    diff_abs = tf.multiply(diff_abs, mask)
    diff_abs_sum = tf.reduce_sum(diff_abs)
    loss_mean = diff_abs_sum / (tf.reduce_sum(mask) * 2 + 1e-6)
    batch_size = tf.to_float(tf.shape(diff)[0])
    loss_sum = diff_abs_sum / batch_size        
    return loss_mean, loss_sum

def epe_loss(diff, mask):
    diff_norm = tf.norm(diff, axis=-1, keepdims=True)
    diff_norm = tf.multiply(diff_norm, mask)
    diff_norm_sum = tf.reduce_sum(diff_norm)
    loss_mean = diff_norm_sum / (tf.reduce_sum(mask) + 1e-6)
    batch_size = tf.to_float(tf.shape(diff)[0])
    loss_sum = diff_norm_sum / batch_size
    return loss_mean, loss_sum


def robust_loss(diff, mask, c=3.0, alpha=1.):
    z_alpha = max(1, 2-alpha)
    diff =  z_alpha / alpha * (tf.pow( tf.square((diff)/c)/z_alpha + 1 , alpha/2.)-1)
    diff = tf.multiply(diff, mask)
    diff_sum = tf.reduce_sum(diff)
    loss_mean = diff_sum / (tf.reduce_sum(mask) * 2 + 1e-6)
    batch_size = tf.to_float(tf.shape(diff)[0])
    loss_sum = diff_sum / batch_size
    return loss_mean, loss_sum

def abs_robust_loss(diff, mask, q=0.4):
    diff = tf.pow((tf.abs(diff)+0.01), q)
    diff = tf.multiply(diff, mask)
    diff_sum = tf.reduce_sum(diff)
    loss_mean = diff_sum / (tf.reduce_sum(mask) * 2 + 1e-6)
    batch_size = tf.to_float(tf.shape(diff)[0])
    loss_sum = diff_sum / batch_size  
    return loss_mean, loss_sum

def compute_losses(flow_gt, flow_estimated, mask, is_scale=True):
    '''
        flow_estimated is a dict, containing flows estimated at different level.
    '''
    flow_gt_level = {}
    mask_level = {}
    
    for i in range(2, 7):
        level = 'level_%d' % i
        flow_size = tf.shape(flow_estimated[level])[1:3]
        flow_gt_level[level] = flow_resize(flow_gt, flow_size, is_scale=is_scale) 
        mask_level[level] = tf.image.resize_images(mask, flow_size, method=1, align_corners=True)
    
    losses = {}
    # --------------------------------- epe ----------------------------------
    epe_mean = {}
    epe_sum = {}
    epe_mean['full_res'], epe_sum['full_res'] = epe_loss(flow_gt - flow_estimated['full_res'], mask)
    epe_mean['refined'], epe_sum['refined'] = epe_loss(flow_gt_level['level_2'] - flow_estimated['refined'], mask_level['level_2'])
    for i in range(2, 7):
        level = 'level_%d' % i   
        epe_mean[level], epe_sum[level] = epe_loss(flow_gt_level[level] - flow_estimated[level], mask_level[level])
    epe_sum['total'] = 0.0025*epe_sum['full_res'] + 0.005*epe_sum['level_2'] + 0.01*epe_sum['level_3'] + \
        0.02*epe_sum['level_4'] + 0.08*epe_sum['level_5'] + 0.32*epe_sum['level_6']   
    losses['epe_mean'] = epe_mean
    losses['epe_sum'] = epe_sum
    
    # --------------------------------- mse ----------------------------------
    mse_mean = {}
    mse_sum = {}
    mse_mean['full_res'], mse_sum['full_res'] = mse_loss(flow_gt - flow_estimated['full_res'], mask)
    mse_mean['refined'], mse_sum['refined'] = mse_loss(flow_gt_level['level_2'] - flow_estimated['refined'], mask_level['level_2'])
    for i in range(2, 7):
        level = 'level_%d' % i   
        mse_mean[level], mse_sum[level] = mse_loss(flow_gt_level[level] - flow_estimated[level], mask_level[level])
    mse_sum['total'] = 0.0025*mse_sum['full_res'] + 0.005*mse_sum['level_2'] + 0.01*mse_sum['level_3'] + \
        0.02*mse_sum['level_4'] + 0.08*mse_sum['level_5'] + 0.32*mse_sum['level_6'] 
    losses['mse_mean'] = mse_mean
    losses['mse_sum'] = mse_sum
        
    # --------------------------------- abs ----------------------------------
    abs_mean = {}
    abs_sum = {}
    abs_mean['full_res'], abs_sum['full_res'] = abs_loss(flow_gt - flow_estimated['full_res'], mask)
    abs_mean['refined'], abs_sum['refined'] = abs_loss(flow_gt_level['level_2'] - flow_estimated['refined'], mask_level['level_2'])
    for i in range(2, 7):
        level = 'level_%d' % i   
        abs_mean[level], abs_sum[level] = abs_loss(flow_gt_level[level] - flow_estimated[level], mask_level[level])
    abs_sum['total'] = 0.0025*abs_sum['full_res'] + 0.005*abs_sum['level_2'] + 0.01*abs_sum['level_3'] + \
        0.02*abs_sum['level_4'] + 0.08*abs_sum['level_5'] + 0.32*abs_sum['level_6'] 
    losses['abs_mean'] = abs_mean
    losses['abs_sum'] = abs_sum
    
    ## -------------------------------- robust ---------------------------------
    #robust_mean = {}
    #robust_sum = {}
    #robust_mean['full_res'], robust_sum['full_res'] = robust_loss(flow_gt - flow_estimated['full_res'], mask)
    #robust_mean['refined'], robust_sum['refined'] = robust_loss(flow_gt_level['level_2'] - flow_estimated['refined'], mask_level['level_2'])
    #for i in range(2, 7):
        #level = 'level_%d' % i   
        #robust_mean[level], robust_sum[level] = robust_loss(flow_gt_level[level] - flow_estimated[level], mask_level[level])
    #robust_sum['total'] = 0.005*robust_sum['refined'] + 0.005*robust_sum['level_2'] + 0.01*robust_sum['level_3'] + \
        #0.02*robust_sum['level_4'] + 0.08*robust_sum['level_5'] + 0.32*robust_sum['level_6'] 
    #losses['robust_mean'] = robust_mean
    #losses['robust_sum'] = robust_sum
        
    ## ------------------------------ abs_robust ---------------------------------
    abs_robust_mean = {}
    abs_robust_sum = {}
    abs_robust_mean['full_res'], abs_robust_sum['full_res'] = abs_robust_loss(flow_gt - flow_estimated['full_res'], mask)
    abs_robust_mean['refined'], abs_robust_sum['refined'] = abs_robust_loss(flow_gt_level['level_2'] - flow_estimated['refined'], mask_level['level_2'])
    for i in range(2, 7):
        level = 'level_%d' % i   
        abs_robust_mean[level], abs_robust_sum[level] = abs_robust_loss(flow_gt_level[level] - flow_estimated[level], mask_level[level])
    abs_robust_sum['total'] = 0.0025*abs_robust_sum['full_res'] + 0.005*abs_robust_sum['level_2'] + 0.01*abs_robust_sum['level_3'] + \
        0.02*abs_robust_sum['level_4'] + 0.08*abs_robust_sum['level_5'] + 0.32*abs_robust_sum['level_6'] 
    losses['abs_robust_mean'] = abs_robust_mean
    losses['abs_robust_sum'] = abs_robust_sum
          
    
    return losses
    
    
