import tensorflow as tf
from tensorflow.contrib import slim
from data_augmentation import flow_resize
from utils import lrelu
from warp import tf_warp

def feature_extractor(x, train=True, trainable=True, reuse=None, regularizer=None, name='feature_extractor'):
    with tf.variable_scope(name, reuse=reuse, regularizer=regularizer):
        with slim.arg_scope([slim.conv2d], activation_fn=lrelu, kernel_size=3, padding='SAME', trainable=trainable):
            net = {}
            net['conv1_1'] = slim.conv2d(x, 16, stride=2, scope='conv1_1')
            net['conv1_2'] = slim.conv2d(net['conv1_1'], 16, stride=1, scope='conv1_2')
            
            net['conv2_1'] = slim.conv2d(net['conv1_2'], 32, stride=2, scope='conv2_1')
            net['conv2_2'] = slim.conv2d(net['conv2_1'], 32, stride=1, scope='conv2_2')
            
            net['conv3_1'] = slim.conv2d(net['conv2_2'], 64, stride=2, scope='conv3_1')
            net['conv3_2'] = slim.conv2d(net['conv3_1'], 64, stride=1, scope='conv3_2')                

            net['conv4_1'] = slim.conv2d(net['conv3_2'], 96, stride=2, scope='conv4_1')
            net['conv4_2'] = slim.conv2d(net['conv4_1'], 96, stride=1, scope='conv4_2')                  
            
            net['conv5_1'] = slim.conv2d(net['conv4_2'], 128, stride=2, scope='conv5_1')
            net['conv5_2'] = slim.conv2d(net['conv5_1'], 128, stride=1, scope='conv5_2') 
            
            net['conv6_1'] = slim.conv2d(net['conv5_2'], 192, stride=2, scope='conv6_1')
            net['conv6_2'] = slim.conv2d(net['conv6_1'], 192, stride=1, scope='conv6_2')  
    
    return net

def context_network(x, flow, train=True, trainable=True, reuse=None, regularizer=None, name='context_network'):
    x_input = tf.concat([x, flow], axis=-1)
    with tf.variable_scope(name, reuse=reuse, regularizer=regularizer):
        with slim.arg_scope([slim.conv2d], activation_fn=lrelu, kernel_size=3, padding='SAME', trainable=trainable):        
            net = {}
            net['dilated_conv1'] = slim.conv2d(x_input, 128, rate=1, scope='dilated_conv1')
            net['dilated_conv2'] = slim.conv2d(net['dilated_conv1'], 128, rate=2, scope='dilated_conv2')
            net['dilated_conv3'] = slim.conv2d(net['dilated_conv2'], 128, rate=4, scope='dilated_conv3')
            net['dilated_conv4'] = slim.conv2d(net['dilated_conv3'], 96, rate=8, scope='dilated_conv4')
            net['dilated_conv5'] = slim.conv2d(net['dilated_conv4'], 64, rate=16, scope='dilated_conv5')
            net['dilated_conv6'] = slim.conv2d(net['dilated_conv5'], 32, rate=1, scope='dilated_conv6')
            net['dilated_conv7'] = slim.conv2d(net['dilated_conv6'], 2, rate=1, activation_fn=None, scope='dilated_conv7')
    
    refined_flow = net['dilated_conv7']
    return refined_flow

def estimator_network(x1, cost_volume, flow, train=True, trainable=True, reuse=None, regularizer=None, name='estimator'):
    net_input = tf.concat([cost_volume, x1, flow], axis=-1)
    with tf.variable_scope(name, reuse=reuse, regularizer=regularizer):
        with slim.arg_scope([slim.conv2d], activation_fn=lrelu, kernel_size=3, padding='SAME', trainable=trainable):        
            net = {}
            net['conv1'] = slim.conv2d(net_input, 128, scope='conv1')
            net['conv2'] = slim.conv2d(net['conv1'], 128, scope='conv2')
            net['conv3'] = slim.conv2d(net['conv2'], 96, scope='conv3')
            net['conv4'] = slim.conv2d(net['conv3'], 64, scope='conv4')
            net['conv5'] = slim.conv2d(net['conv4'], 32, scope='conv5')
            net['conv6'] = slim.conv2d(net['conv5'], 2, activation_fn=None, scope='conv6')
    
    #flow_estimated = net['conv6']
    
    return net

def compute_cost_volume(x1, x2, H, W, channel, d=9):
    x1 = tf.nn.l2_normalize(x1, axis=3)
    x2 = tf.nn.l2_normalize(x2, axis=3)        

    # choice 1: use tf.extract_image_patches, may not work for some tensorflow versions
    x2_patches = tf.extract_image_patches(x2, [1, d, d, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
    
    # choice 2: use convolution, but is slower than choice 1
    # out_channels = d * d
    # w = tf.eye(out_channels*channel, dtype=tf.float32)
    # w = tf.reshape(w, (d, d, channel, out_channels*channel))
    # x2_patches = tf.nn.conv2d(x2, w, strides=[1, 1, 1, 1], padding='SAME')
    
    x2_patches = tf.reshape(x2_patches, [-1, H, W, d, d, channel])
    x1_reshape = tf.reshape(x1, [-1, H, W, 1, 1, channel])
    x1_dot_x2 = tf.multiply(x1_reshape, x2_patches)
    
    cost_volume = tf.reduce_sum(x1_dot_x2, axis=-1)
    #cost_volume = tf.reduce_mean(x1_dot_x2, axis=-1)
    cost_volume = tf.reshape(cost_volume, [-1, H, W, d*d])   
    return cost_volume

def estimator(x0, x1, x2, flow_fw, flow_bw, train=True, trainable=True, reuse=None, regularizer=None, name='estimator'):           
    # warp x2 according to flow
    if train:
        x_shape = x1.get_shape().as_list()
    else:
        x_shape = tf.shape(x1)
    H = x_shape[1]
    W = x_shape[2]
    channel = x_shape[3]
    x2_warp = tf_warp(x2, flow_fw, H, W)
    x0_warp = tf_warp(x0, flow_bw, H, W)
    
    # ---------------cost volume-----------------
    
    cost_volume_fw = compute_cost_volume(x1, x2_warp, H, W, channel, d=9)
    cost_volume_bw = compute_cost_volume(x1, x0_warp, H, W, channel, d=9)
    
    cv_concat_fw = tf.concat([cost_volume_fw, cost_volume_bw], -1)
    cv_concat_bw = tf.concat([cost_volume_bw, cost_volume_fw], -1)
    
    flow_concat_fw = tf.concat([flow_fw, -flow_bw], -1)
    flow_concat_bw = tf.concat([flow_bw, -flow_fw], -1)
  
    net_fw = estimator_network(x1, cv_concat_fw, flow_concat_fw, train=train, trainable=trainable, reuse=reuse, regularizer=regularizer, name=name)
    net_bw = estimator_network(x1, cv_concat_bw, flow_concat_bw, train=train, trainable=trainable, reuse=True, regularizer=regularizer, name=name)
    
    return net_fw, net_bw

def pyramid_processing_three_frame(batch_img, x0_feature, x1_feature, x2_feature, train=True, trainable=True, reuse=None, regularizer=None, is_scale=True):
    x_shape = tf.shape(x1_feature['conv6_2'])
    initial_flow_fw = tf.zeros([x_shape[0], x_shape[1], x_shape[2], 2], dtype=tf.float32, name='initial_flow_fw')
    initial_flow_bw = tf.zeros([x_shape[0], x_shape[1], x_shape[2], 2], dtype=tf.float32, name='initial_flow_bw')
    flow_fw = {}
    flow_bw = {}
    net_fw, net_bw = estimator(x0_feature['conv6_2'], x1_feature['conv6_2'], x2_feature['conv6_2'], 
        initial_flow_fw, initial_flow_bw, train=train, trainable=trainable, reuse=reuse, regularizer=regularizer, name='estimator_level_6')
    flow_fw['level_6'] = net_fw['conv6']
    flow_bw['level_6'] = net_bw['conv6']
    
    
    for i in range(4):
        feature_name = 'conv%d_2' % (5-i)
        level = 'level_%d' % (5-i)
        feature_size = tf.shape(x1_feature[feature_name])[1:3]
        initial_flow_fw = flow_resize(flow_fw['level_%d' % (6-i)], feature_size, is_scale=is_scale)
        initial_flow_bw = flow_resize(flow_bw['level_%d' % (6-i)], feature_size, is_scale=is_scale)
        net_fw, net_bw = estimator(x0_feature[feature_name], x1_feature[feature_name], x2_feature[feature_name], 
            initial_flow_fw, initial_flow_bw, train=train, trainable=trainable, reuse=reuse, regularizer=regularizer, name='estimator_level_%d' % (5-i))           
        flow_fw[level] = net_fw['conv6']
        flow_bw[level] = net_bw['conv6'] 
    
    flow_concat_fw = tf.concat([flow_fw['level_2'], -flow_bw['level_2']], -1)
    flow_concat_bw = tf.concat([flow_bw['level_2'], -flow_fw['level_2']], -1)       
    
    x_feature = tf.concat([net_fw['conv5'], net_bw['conv5']], axis=-1)
    flow_fw['refined'] = context_network(x_feature, flow_concat_fw, train=train, trainable=trainable, reuse=reuse, regularizer=regularizer, name='context_network')
    flow_size = tf.shape(batch_img)[1:3]
    flow_fw['full_res'] = flow_resize(flow_fw['refined'], flow_size, is_scale=is_scale) 
    
    x_feature = tf.concat([net_bw['conv5'], net_fw['conv5']], axis=-1)
    flow_bw['refined'] = context_network(x_feature, flow_concat_bw, train=train, trainable=trainable, reuse=True, regularizer=regularizer, name='context_network')
    flow_bw['full_res'] = flow_resize(flow_bw['refined'], flow_size, is_scale=is_scale)  
    
    return flow_fw, flow_bw


def pyramid_processing(batch_img0, batch_img1, batch_img2, train=True, trainable=True, reuse=None, regularizer=None, is_scale=True):
    x0_feature = feature_extractor(batch_img0, train=train, trainable=trainable, regularizer=regularizer, name='feature_extractor')
    x1_feature = feature_extractor(batch_img1, train=train, trainable=trainable, reuse=True, regularizer=regularizer, name='feature_extractor')
    x2_feature = feature_extractor(batch_img2, train=train, trainable=trainable, reuse=True, regularizer=regularizer, name='feature_extractor')
    
    flow_fw, flow_bw = pyramid_processing_three_frame(batch_img0, x0_feature, x1_feature, x2_feature, train=train, trainable=trainable, 
                                                      reuse=reuse, regularizer=regularizer, is_scale=is_scale)   
    return flow_fw, flow_bw


def pyramid_processing_five_frame(batch_img0, batch_img1, batch_img2, batch_img3, batch_img4, train=True, trainable=True, regularizer=None, is_scale=True):
    x0_feature = feature_extractor(batch_img0, train=train, trainable=trainable, regularizer=regularizer, name='feature_extractor')
    x1_feature = feature_extractor(batch_img1, train=train, trainable=trainable, reuse=True, regularizer=regularizer, name='feature_extractor')
    x2_feature = feature_extractor(batch_img2, train=train, trainable=trainable, reuse=True, regularizer=regularizer, name='feature_extractor')
    x3_feature = feature_extractor(batch_img3, train=train, trainable=trainable, reuse=True, regularizer=regularizer, name='feature_extractor')
    x4_feature = feature_extractor(batch_img4, train=train, trainable=trainable, reuse=True, regularizer=regularizer, name='feature_extractor') 
    
    flow_fw_12, flow_bw_10 = pyramid_processing_three_frame(batch_img0, x0_feature, x1_feature, x2_feature, train=train, trainable=trainable, reuse=None, regularizer=regularizer, is_scale=is_scale)
    flow_fw_23, flow_bw_21 = pyramid_processing_three_frame(batch_img0, x1_feature, x2_feature, x3_feature, train=train, trainable=trainable, reuse=True, regularizer=regularizer, is_scale=is_scale)
    flow_fw_34, flow_bw_32 = pyramid_processing_three_frame(batch_img0, x2_feature, x3_feature, x4_feature, train=train, trainable=trainable, reuse=True, regularizer=regularizer, is_scale=is_scale)
        
    return flow_fw_12, flow_bw_10, flow_fw_23, flow_bw_21, flow_fw_34, flow_bw_32
