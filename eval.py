import tensorflow as tf
import numpy as np
import os
import time
import cv2
from scipy import misc, io

from pwc_network import pyramid_processing, pyramid_processing_five_frame
from warp import tf_warp
from config.extract_config import config_dict
from datasets import KittiDataset, SintelDataset, KittiRawDataset, SintelRawDataset
from data_augmentation import flow_resize
from flowlib import flow_to_color, flow_error_image
from losses import mse_loss, epe_loss, abs_loss
from utils import compute_Fl, occlusion, length_sq, rgb_bgr

# manually select one free gpu to train
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#autonatically select one free gpu to train
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
os.system('rm tmp')

def create_dataset_and_iterator(dataset_name='Kitti', dataset_config={}, num_parallel_calls=4):
    if dataset_name == 'Kitti':
        dataset = KittiDataset(data_list_file=dataset_config['data_list_file'], 
                               img_dir=dataset_config['img_dir'])
    elif dataset_name == 'Sintel':
        dataset = SintelDataset(data_list_file=dataset_config['data_list_file'], 
                                img_dir=dataset_config['img_dir'],
                                dataset_type=dataset_config['dataset_type'])
    else:
        raise ValueError('Invalid dataset. Dataset should be one of {FlyingChairs, FlyingThings3D, Kitti, Sintel}')        
    
    iterator = dataset.create_one_shot_iterator(dataset.data_list, num_parallel_calls=num_parallel_calls)
    return dataset, iterator
        
def generate_batch_input(dataset_name, iterator):
    if dataset_name in ['Kitti', 'Sintel']:
        batch_img0_raw, batch_img1_raw, batch_img2_raw, batch_img0, batch_img1, batch_img2, batch_flow, batch_mask = iterator.get_next()
    else:
        raise ValueError('Invalid dataset. Dataset should be one of {Kitti, Sintel}')
    return batch_img0_raw, batch_img1_raw, batch_img2_raw, batch_img0, batch_img1, batch_img2, batch_flow, batch_mask  

def test_kitti_2015(restore_model_dir, model_name, start_step, end_step, checkpoint_interval, dataset_config={}, is_scale=True, is_write_summmary=True, num_parallel_calls=4, network_mode='v1'):
    dataset_name = 'Kitti'
    dataset = KittiDataset(data_list_file=dataset_config['data_list_file'], img_dir=dataset_config['img_dir'])        
    iterator = dataset.create_evaluation_2015_iterator(dataset.data_list, num_parallel_calls)
    batch_img0, batch_img1, batch_img2, batch_flow, batch_mask, batch_noc_mask, batch_fg_mask = iterator.get_next()
    batch_occ_mask = batch_mask - batch_noc_mask
    batch_occ_mask = tf.where(batch_occ_mask < 0., tf.zeros_like(batch_occ_mask), batch_occ_mask)
    batch_bg_mask = batch_mask - batch_fg_mask
    batch_noc_fg_mask = tf.multiply(batch_fg_mask, batch_noc_mask)
    batch_noc_bg_mask = batch_noc_mask - batch_noc_fg_mask
    
    data_num = dataset.data_num
    flow_estimated, _ = pyramid_processing(batch_img0, batch_img1, batch_img2, train=False, trainable=False, is_scale=is_scale, network_mode=network_mode)
    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)    
    
    diff = flow_estimated['full_res'] - batch_flow
    EPE_loss, _ = epe_loss(diff, batch_mask)
    EPE_loss_noc, _ = epe_loss(diff, batch_noc_mask)
    EPE_loss_occ, _ = epe_loss(diff, batch_occ_mask)    
    MSE_loss, _ = mse_loss(diff, batch_mask)
    ABS_loss, _ = abs_loss(diff, batch_mask)
    Fl_all = compute_Fl(batch_flow, flow_estimated['full_res'], batch_mask)
    Fl_occ = compute_Fl(batch_flow, flow_estimated['full_res'], batch_occ_mask)
    Fl_noc = compute_Fl(batch_flow, flow_estimated['full_res'], batch_noc_mask)
    Fl_fg_all = compute_Fl(batch_flow, flow_estimated['full_res'], batch_fg_mask)
    Fl_bg_all = compute_Fl(batch_flow, flow_estimated['full_res'], batch_bg_mask)
    Fl_fg_noc = compute_Fl(batch_flow, flow_estimated['full_res'], batch_noc_fg_mask)
    Fl_bg_noc = compute_Fl(batch_flow, flow_estimated['full_res'], batch_noc_bg_mask)
    mask_num = tf.reduce_sum(batch_mask)
    noc_mask_num = tf.reduce_sum(batch_noc_mask)
    occ_mask_num = tf.reduce_sum(batch_occ_mask)    
    
    summary_writer = tf.summary.FileWriter(logdir='/'.join([dataset_name, 'summary', 'test', model_name]))
    saver = tf.train.Saver(var_list=restore_vars)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True     
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer()) 
    sess.run(iterator.initializer)
    steps = np.arange(start_step, end_step+1, checkpoint_interval, dtype='int32')
    for step in steps:
        saver.restore(sess, '%s/%s/model-%d' % (restore_model_dir, model_name, step)) 
        
        EPE = np.zeros([data_num])
        EPE_noc = np.zeros([data_num])
        EPE_occ = np.zeros([data_num])        
        Fl = np.zeros([data_num])     
        Fl_noc_value = np.zeros([data_num])
        Fl_occ_value = np.zeros([data_num])        
        Fl_fg_all_value = np.zeros([data_num])
        Fl_bg_all_value = np.zeros([data_num])
        Fl_fg_noc_value = np.zeros([data_num])
        Fl_bg_noc_value = np.zeros([data_num])
        MASK_NUM = np.zeros([data_num])
        NOC_MASK_NUM = np.zeros([data_num])
        OCC_MASK_NUM = np.zeros([data_num])           
        
        start_time = time.time()
        for i in range(data_num):
            EPE[i], EPE_noc[i], EPE_occ[i], Fl[i], Fl_noc_value[i], Fl_occ_value[i], Fl_fg_all_value[i], Fl_bg_all_value[i], Fl_fg_noc_value[i], Fl_bg_noc_value[i], MASK_NUM[i], NOC_MASK_NUM[i], OCC_MASK_NUM[i] = sess.run(
                [EPE_loss, EPE_loss_noc, EPE_loss_occ, Fl_all, Fl_noc, Fl_occ, Fl_fg_all, Fl_bg_all, Fl_fg_noc, Fl_bg_noc, mask_num, noc_mask_num, occ_mask_num])   

        mean_time = (time.time()-start_time) / data_num 
        
        mean_EPE = np.mean(EPE)
        mean_EPE_noc = np.mean(EPE_noc)
        mean_EPE_occ = np.mean(EPE_occ)
        mean_Fl = np.mean(Fl)
        mean_Fl_noc = np.mean(Fl_noc_value)
        mean_Fl_occ = np.mean(Fl_occ_value)
        mean_Fl_fg_all = np.mean(Fl_fg_all_value)
        mean_Fl_bg_all = np.mean(Fl_bg_all_value)
        mean_Fl_fg_noc = np.mean(Fl_fg_noc_value)
        mean_Fl_bg_noc = np.mean(Fl_bg_noc_value)        
    
        mean_weighted_EPE = np.sum(np.multiply(EPE, MASK_NUM)) / np.sum(MASK_NUM)
        mean_weighted_EPE_noc = np.sum(np.multiply(EPE_noc, NOC_MASK_NUM)) / np.sum(NOC_MASK_NUM)
        mean_weighted_EPE_occ = np.sum(np.multiply(EPE_occ, OCC_MASK_NUM)) / np.sum(OCC_MASK_NUM)
        mean_weighted_Fl = np.sum(np.multiply(Fl, MASK_NUM)) / np.sum(MASK_NUM)
        mean_weighted_Fl_noc = np.sum(np.multiply(Fl_noc_value, NOC_MASK_NUM)) / np.sum(NOC_MASK_NUM)
        mean_weighted_Fl_occ = np.sum(np.multiply(Fl_occ_value, OCC_MASK_NUM)) / np.sum(OCC_MASK_NUM)
        
        print('step %d: EPE: %.6f, EPE_noc: %.6f, EPE_occ: %.6f, Fl_all: %.6f, Fl_noc: %.6f, Fl_occ: %.6f, Fl_fg_all: %.6f, Fl_bg_all: %.6f, Fl_fg_noc: %.6f, Fl_bg_noc: %.6f' % 
              (step, mean_EPE, mean_EPE_noc, mean_EPE_occ, mean_Fl, mean_Fl_noc, mean_Fl_occ, mean_Fl_fg_all, mean_Fl_bg_all, mean_Fl_fg_noc, mean_Fl_bg_noc))
        print('      weighted_EPE: %.6f, weighted_EPE_noc: %.6f, weighted_EPE_occ: %.6f, weighted_Fl: %.6f, weighed_Fl_noc: %.6f, weighted_Fl_occ: %.6f, mean_time: %.6f, occ_num: %f' %
              (mean_weighted_EPE, mean_weighted_EPE_noc, mean_weighted_EPE_occ, mean_weighted_Fl, mean_weighted_Fl_noc, mean_weighted_Fl_occ, mean_time, np.sum(OCC_MASK_NUM)))

        if is_write_summmary:
            summary = tf.Summary()
            summary.value.add(tag='EPE', simple_value=mean_EPE)
            summary.value.add(tag='EPE_noc', simple_value=mean_EPE_noc)
            summary.value.add(tag='EPE_occ', simple_value=mean_EPE_occ)
            summary.value.add(tag='mse', simple_value=mean_MSE)
            summary.value.add(tag='abs', simple_value=mean_ABS)
            summary.value.add(tag='Fl', simple_value=mean_Fl)
            summary.value.add(tag='Fl_noc', simple_value=mean_Fl_noc)
            summary.value.add(tag='Fl_occ', simple_value=mean_Fl_occ)
            summary.value.add(tag='Fl_fg_all', simple_value=mean_Fl_fg_all)
            summary.value.add(tag='Fl_bg_all', simple_value=mean_Fl_bg_all)
            summary.value.add(tag='Fl_fg_noc', simple_value=mean_Fl_fg_noc)
            summary.value.add(tag='Fl_bg_noc', simple_value=mean_Fl_bg_noc)
            summary.value.add(tag='weighted/EPE', simple_value=mean_weighted_EPE)
            summary.value.add(tag='weighted/EPE_noc', simple_value=mean_weighted_EPE_noc)
            summary.value.add(tag='weighted/EPE_occ', simple_value=mean_weighted_EPE_occ)
            summary.value.add(tag='weighted/Fl', simple_value=mean_weighted_Fl)
            summary.value.add(tag='weighted/Fl_noc', simple_value=mean_weighted_Fl_noc)
            summary.value.add(tag='weighted/Fl_occ', simple_value=mean_weighted_Fl_occ)
            summary.value.add(tag='time_cost', simple_value=mean_time)
            summary_writer.add_summary(summary, global_step=step)
            
def test_kitti_2012(restore_model_dir, model_name, start_step, end_step, checkpoint_interval, dataset_config={}, is_scale=True, is_write_summmary=True, num_parallel_calls=4, network_mode='v2_concat'):
    dataset_name = 'Kitti'
    dataset = KittiDataset(data_list_file=dataset_config['data_list_file'], img_dir=dataset_config['img_dir'])        
    iterator = dataset.create_evaluation_2012_iterator(dataset.data_list, num_parallel_calls)
    batch_img0, batch_img1, batch_img2, batch_flow, batch_mask, batch_noc_mask = iterator.get_next()
    batch_occ_mask = batch_mask - batch_noc_mask
    batch_occ_mask = tf.where(batch_occ_mask < 0., tf.zeros_like(batch_occ_mask), batch_occ_mask)
    
    data_num = dataset.data_num
    
    flow_estimated, _ = pyramid_processing(batch_img0, batch_img1, batch_img2, train=False, trainable=False, is_scale=is_scale, network_mode=network_mode)
    
    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)    
    
    diff = flow_estimated['full_res'] - batch_flow
    EPE_loss, _ = epe_loss(diff, batch_mask)
    EPE_loss_noc, _ = epe_loss(diff, batch_noc_mask)
    EPE_loss_occ, _ = epe_loss(diff, batch_occ_mask)
    MSE_loss, _ = mse_loss(diff, batch_mask)
    ABS_loss, _ = abs_loss(diff, batch_mask)
    Fl_all = compute_Fl(batch_flow, flow_estimated['full_res'], batch_mask)
    Fl_noc = compute_Fl(batch_flow, flow_estimated['full_res'], batch_noc_mask)
    Fl_occ = compute_Fl(batch_flow, flow_estimated['full_res'], batch_occ_mask)
    mask_num = tf.reduce_sum(batch_mask)
    noc_mask_num = tf.reduce_sum(batch_noc_mask)
    occ_mask_num = tf.reduce_sum(batch_occ_mask)
    
    summary_writer = tf.summary.FileWriter(logdir='/'.join([dataset_name, 'summary', 'test', model_name]))
    saver = tf.train.Saver(var_list=restore_vars)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True     
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer()) 
    sess.run(iterator.initializer)
    steps = np.arange(start_step, end_step+1, checkpoint_interval, dtype='int32')
    for step in steps:
        saver.restore(sess, '%s/%s/model-%d' % (restore_model_dir, model_name, step)) 
        
        EPE = np.zeros([data_num])
        EPE_noc = np.zeros([data_num])
        EPE_occ = np.zeros([data_num])
        MSE = np.zeros([data_num])
        ABS = np.zeros([data_num])
        Fl = np.zeros([data_num])
        Fl_noc_value = np.zeros([data_num])
        Fl_occ_value = np.zeros([data_num])
        MASK_NUM = np.zeros([data_num])
        NOC_MASK_NUM = np.zeros([data_num])
        OCC_MASK_NUM = np.zeros([data_num])  
        
        start_time = time.time()
        for i in range(data_num):
            EPE[i], EPE_noc[i], EPE_occ[i], MSE[i], ABS[i], Fl[i], Fl_noc_value[i], Fl_occ_value[i], MASK_NUM[i], NOC_MASK_NUM[i], OCC_MASK_NUM[i] = sess.run(
                [EPE_loss, EPE_loss_noc, EPE_loss_occ, MSE_loss, ABS_loss, Fl_all, Fl_noc, Fl_occ, mask_num, noc_mask_num, occ_mask_num])  
        
        mean_time = (time.time()-start_time) / data_num
        mean_Fl = np.mean(Fl)
        mean_Fl_noc = np.mean(Fl_noc_value)
        mean_Fl_occ = np.mean(Fl_occ_value)
         
        mean_EPE = np.mean(EPE)
        mean_EPE_noc = np.mean(EPE_noc)
        mean_EPE_occ = np.mean(EPE_occ)
        mean_MSE = np.mean(MSE)
        mean_ABS = np.mean(ABS)
        
        mean_weighted_EPE = np.sum(np.multiply(EPE, MASK_NUM)) / np.sum(MASK_NUM)
        mean_weighted_EPE_noc = np.sum(np.multiply(EPE_noc, NOC_MASK_NUM)) / np.sum(NOC_MASK_NUM)
        mean_weighted_EPE_occ = np.sum(np.multiply(EPE_occ, OCC_MASK_NUM)) / np.sum(OCC_MASK_NUM)
        mean_weighted_Fl = np.sum(np.multiply(Fl, MASK_NUM)) / np.sum(MASK_NUM)
        mean_weighted_Fl_noc = np.sum(np.multiply(Fl_noc_value, NOC_MASK_NUM)) / np.sum(NOC_MASK_NUM)
        mean_weighted_Fl_occ = np.sum(np.multiply(Fl_occ_value, OCC_MASK_NUM)) / np.sum(OCC_MASK_NUM)
        
        print('step %d: EPE: %.6f, EPE_noc: %.6f, EPE_occ: %.6f, Fl_all: %.6f, Fl_noc: %.6f, Fl_occ: %.6f, weighted_EPE: %.6f, weighted_EPE_noc: %.6f, weighted_EPE_occ: %.6f, weighted_Fl: %.6f, weighed_Fl_noc: %.6f, weighted_Fl_occ: %.6f, time_cost: %.6f' % 
              (step, mean_EPE, mean_EPE_noc, mean_EPE_occ, mean_Fl, mean_Fl_noc, mean_Fl_occ, mean_weighted_EPE, mean_weighted_EPE_noc, mean_weighted_EPE_occ, mean_weighted_Fl, mean_weighted_Fl_noc, mean_weighted_Fl_occ, mean_time))

        if is_write_summmary:
            summary = tf.Summary()
            summary.value.add(tag='time_cost', simple_value=mean_time)
            summary.value.add(tag='EPE', simple_value=mean_EPE)
            summary.value.add(tag='EPE_noc', simple_value=mean_EPE_noc)
            summary.value.add(tag='EPE_occ', simple_value=mean_EPE_occ)
            summary.value.add(tag='mse', simple_value=mean_MSE)
            summary.value.add(tag='abs', simple_value=mean_ABS)
            summary.value.add(tag='Fl', simple_value=mean_Fl)
            summary.value.add(tag='Fl_noc', simple_value=mean_Fl_noc)
            summary.value.add(tag='Fl_occ', simple_value=mean_Fl_occ)
            summary.value.add(tag='weighted_EPE', simple_value=mean_weighted_EPE)
            summary.value.add(tag='weighted_EPE_noc', simple_value=mean_weighted_EPE_noc)
            summary.value.add(tag='weighted_EPE_occ', simple_value=mean_weighted_EPE_occ)
            summary.value.add(tag='weighted_Fl', simple_value=mean_weighted_Fl)
            summary.value.add(tag='weighted_Fl_noc', simple_value=mean_weighted_Fl_noc)
            summary.value.add(tag='weighted_Fl_occ', simple_value=mean_weighted_Fl_occ)
            summary_writer.add_summary(summary, global_step=step)

def test_sintel(restore_model_dir, model_name, start_step, end_step, checkpoint_interval, dataset_config={}, is_scale=True, is_write_summmary=True, num_parallel_calls=4, network_mode='v1'):
    dataset_name = 'Sintel'
    dataset = SintelDataset(data_list_file=dataset_config['data_list_file'], 
                            img_dir=dataset_config['img_dir'],
                            dataset_type=dataset_config['dataset_type'])    
    iterator = dataset.create_evaluation_iterator(dataset.data_list, num_parallel_calls=num_parallel_calls)
    batch_img0, batch_img1, batch_img2, batch_flow, batch_mask, batch_occlusion = iterator.get_next()    
    batch_mask_occ = tf.multiply(batch_occlusion, batch_mask)
    batch_mask_noc = tf.multiply(1-batch_occlusion, batch_mask)
    batch_flow_norm = tf.norm(batch_flow, axis=-1, keepdims=True)
    batch_mask_s0_10 = tf.cast(tf.logical_and(batch_flow_norm >= 0., batch_flow_norm < 10.), tf.float32)
    batch_mask_s0_10 = tf.multiply(batch_mask_s0_10, batch_mask)
    batch_mask_s10_40 = tf.cast(tf.logical_and(batch_flow_norm >= 10., batch_flow_norm < 40.), tf.float32)
    batch_mask_s10_40 = tf.multiply(batch_mask_s10_40, batch_mask)
    batch_mask_s40_plus = tf.cast(batch_flow_norm >= 40., tf.float32)
    batch_mask_s40_plus = tf.multiply(batch_mask_s40_plus, batch_mask)
    
    num_mask = tf.reduce_sum(batch_mask)
    num_mask_occ = tf.reduce_sum(batch_mask_occ)
    num_mask_noc = tf.reduce_sum(batch_mask_noc)
    num_mask_s0_10 = tf.reduce_sum(batch_mask_s0_10)
    num_mask_s10_40 = tf.reduce_sum(batch_mask_s10_40)
    num_mask_s40_plus = tf.reduce_sum(batch_mask_s40_plus)
    
    data_num = dataset.data_num
    
    flow_estimated, _ = pyramid_processing(batch_img0, batch_img1, batch_img2, train=False, trainable=False, is_scale=is_scale, network_mode=network_mode)
    
    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)    
    diff = flow_estimated['full_res'] - batch_flow
    EPE_loss, _ = epe_loss(diff, batch_mask)
    MSE_loss, _ = mse_loss(diff, batch_mask)
    ABS_loss, _ = abs_loss(diff, batch_mask)
    Fl_all = compute_Fl(batch_flow, flow_estimated['full_res'], batch_mask)
    EPE_loss_matched, _ = epe_loss(diff, batch_mask_noc)
    EPE_loss_matched = tf.where(tf.reduce_sum(batch_mask_noc) < 1., 0., EPE_loss_matched)
    EPE_loss_unmatched, _ = epe_loss(diff, batch_mask_occ)   
    EPE_loss_unmatched = tf.where(tf.reduce_sum(batch_mask_occ) < 1., 0., EPE_loss_unmatched)
    EPE_loss_s0_10, _ = epe_loss(diff, batch_mask_s0_10)
    EPE_loss_s0_10 = tf.where(tf.reduce_sum(batch_mask_s0_10) < 1., 0., EPE_loss_s0_10)
    EPE_loss_s10_40, _ = epe_loss(diff, batch_mask_s10_40)
    EPE_loss_s10_40 = tf.where(tf.reduce_sum(batch_mask_s10_40) < 1., 0., EPE_loss_s10_40)
    EPE_loss_s40_plus, _ = epe_loss(diff, batch_mask_s40_plus)
    EPE_loss_s40_plus = tf.where(tf.reduce_sum(batch_mask_s40_plus) < 1, 0., EPE_loss_s40_plus)
    

    summary_writer = tf.summary.FileWriter(logdir='/'.join([dataset_name, 'summary', 'test', model_name]))
    saver = tf.train.Saver(var_list=restore_vars)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True     
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer()) 
    sess.run(iterator.initializer)
    steps = np.arange(start_step, end_step+1, checkpoint_interval, dtype='int32')
    for step in steps:
        saver.restore(sess, '%s/%s/model-%d' % (restore_model_dir, model_name, step))   
        EPE = np.zeros([data_num])
        MSE = np.zeros([data_num])
        ABS = np.zeros([data_num])
        Fl = np.zeros([data_num])
        EPE_matched = np.zeros([data_num])
        EPE_unmatched = np.zeros([data_num])
        EPE_s0_10 = np.zeros([data_num])
        EPE_s10_40 = np.zeros([data_num])
        EPE_s40_plus = np.zeros([data_num])
        
        np_num_mask = np.zeros([data_num])
        np_num_mask_occ = np.zeros([data_num])
        np_num_mask_noc = np.zeros([data_num])
        np_num_mask_s0_10 = np.zeros([data_num])
        np_num_mask_s10_40 = np.zeros([data_num])
        np_num_mask_s40_plus = np.zeros([data_num])
        start_time = time.time()
        for i in range(data_num):
            EPE[i], MSE[i], ABS[i], Fl[i], EPE_matched[i], EPE_unmatched[i], EPE_s0_10[i], EPE_s10_40[i], EPE_s40_plus[i], \
                np_num_mask[i], np_num_mask_occ[i], np_num_mask_noc[i], np_num_mask_s0_10[i], np_num_mask_s10_40[i], np_num_mask_s40_plus[i] = sess.run(
                [EPE_loss, MSE_loss, ABS_loss, Fl_all, EPE_loss_matched, EPE_loss_unmatched, EPE_loss_s0_10, EPE_loss_s10_40, EPE_loss_s40_plus, 
                 num_mask, num_mask_occ, num_mask_noc, num_mask_s0_10, num_mask_s10_40, num_mask_s40_plus])     
        
        mean_time = (time.time()-start_time) / data_num 
        mean_EPE = np.mean(EPE)
        mean_MSE = np.mean(MSE)
        mean_ABS = np.mean(ABS)
        mean_Fl = np.mean(Fl) 
        mean_EPE_matched = np.mean(EPE_matched)
        mean_EPE_unmatched = np.mean(EPE_unmatched)
        mean_EPE_s0_10 = np.mean(EPE_s0_10)
        mean_EPE_s10_40 = np.mean(EPE_s10_40)
        mean_EPE_s40_plus = np.mean(EPE_s40_plus)
        
        weighted_mean_EPE = np.sum(np.multiply(EPE, np_num_mask)) / np.sum(np_num_mask)
        weighted_mean_MSE = np.sum(np.multiply(MSE, np_num_mask)) / np.sum(np_num_mask)
        weighted_mean_ABS = np.sum(np.multiply(ABS, np_num_mask)) / np.sum(np_num_mask)
        weighted_mean_Fl = np.sum(np.multiply(Fl, np_num_mask)) / np.sum(np_num_mask)
        weighted_mean_EPE_matched = np.sum(np.multiply(EPE_matched, np_num_mask_noc)) / np.sum(np_num_mask_noc)
        weighted_mean_EPE_unmatched = np.sum(np.multiply(EPE_unmatched, np_num_mask_occ)) / np.sum(np_num_mask_occ)
        weighted_mean_EPE_s0_10 = np.sum(np.multiply(EPE_s0_10, np_num_mask_s0_10)) / np.sum(np_num_mask_s0_10)
        weighted_mean_EPE_s10_40 = np.sum(np.multiply(EPE_s10_40, np_num_mask_s10_40)) / np.sum(np_num_mask_s10_40)
        weighted_mean_EPE_s40_plus = np.sum(np.multiply(EPE_s40_plus, np_num_mask_s40_plus)) / np.sum(np_num_mask_s40_plus)       

        print('step %d: EPE: %.6f, mse: %.6f, abs: %.6f, Fl: %.6f, EPE_matched: %.6f, EPE_unmatched: %.6f, EPE_s0_10: %.6f, EPE_s10_40: %.6f, EPE_s40_plus: %.6f, \n \
               weighted_EPE: %.6f, weighted_MSE: %.6f, weighted_ABS: %.6f, weighted_Fl: %.6f, weighted_EPE_matched: %.6f, weighted_EPE_unmatched: %.6f, \n \
               weighted_EPE_s0_10: %.6f, weighted_EPE_s10_40: %.6f, weighted_EPE_s40_plus: %.6f, time_cost: %.6f' % 
              (step, mean_EPE, mean_MSE, mean_ABS, mean_Fl, mean_EPE_matched, mean_EPE_unmatched, mean_EPE_s0_10, mean_EPE_s10_40, mean_EPE_s40_plus, 
               weighted_mean_EPE, weighted_mean_MSE, weighted_mean_ABS, weighted_mean_Fl, weighted_mean_EPE_matched, weighted_mean_EPE_unmatched, 
               weighted_mean_EPE_s0_10, weighted_mean_EPE_s10_40, weighted_mean_EPE_s40_plus, mean_time))
        
        #print('step %d: EPE: %.6f, mse: %.6f, abs: %.6f, Fl: %.6f, EPE_matched: %.6f, EPE_unmatched: %.6f, EPE_s0_10: %.6f, EPE_s10_40: %.6f, EPE_s40_plus: %.6f, time_cost: %.6f' % 
                      #(step, mean_EPE, mean_MSE, mean_ABS, mean_Fl, mean_EPE_matched, mean_EPE_unmatched, mean_EPE_s0_10, mean_EPE_s10_40, mean_EPE_s40_plus, mean_time))        
        
        if is_write_summmary:
            summary = tf.Summary()
            summary.value.add(tag='EPE', simple_value=mean_EPE)
            summary.value.add(tag='mse', simple_value=mean_MSE)
            summary.value.add(tag='abs', simple_value=mean_ABS)
            summary.value.add(tag='Fl', simple_value=mean_Fl)
            summary.value.add(tag='EPE_matched', simple_value=mean_EPE_matched)
            summary.value.add(tag='EPE_unmatched', simple_value=mean_EPE_unmatched)
            summary.value.add(tag='EPE_s0_10', simple_value=mean_EPE_s0_10)
            summary.value.add(tag='EPE_s10_40', simple_value=mean_EPE_s10_40)
            summary.value.add(tag='EPE_s40_plus', simple_value=mean_EPE_s40_plus)
            summary.value.add(tag='time_cost', simple_value=mean_time)  
            summary_writer.add_summary(summary, global_step=step)
            
def test(dataset_name, restore_model_dir, model_name, start_step, end_step, checkpoint_interval, dataset_config={}, is_scale=True, is_write_summmary=True, num_parallel_calls=4, network_mode='v1'):
    if dataset_name == 'Kitti':
        test_kitti_2015(restore_model_dir, model_name, start_step, end_step, checkpoint_interval, dataset_config, is_scale, is_write_summmary, num_parallel_calls, network_mode=network_mode)
    elif dataset_name == 'Sintel':
        test_sintel(restore_model_dir, model_name, start_step, end_step, checkpoint_interval, dataset_config, is_scale, is_write_summmary, num_parallel_calls, network_mode=network_mode)
    else:
        raise ValueError('Invalid dataset. Dataset should be one of {Kitti, Sintel}')      
        

def kitti_raw_prediction(dataset_name, restore_model_dir, model_name, sample_step=100000, dataset_config={}, is_scale=True, num_parallel_calls=4, network_mode='v1'):
    dataset = KittiRawDataset(data_list_file=dataset_config['data_list_file'], img_dir=dataset_config['img_dir'])
    data_num = dataset.data_num
    iterator = dataset.create_prediction_iterator(dataset.data_list, num_parallel_calls=num_parallel_calls)
    batch_img0, batch_img1, batch_img2, batch_img3, batch_img4 = iterator.get_next()
    flow_fw_12, flow_bw_10, flow_fw_23, flow_bw_21, flow_fw_34, flow_bw_32 = pyramid_processing_five_frame(batch_img0, batch_img1, batch_img2, batch_img3, batch_img4, 
                                                                                                           train=False, trainable=False, regularizer=None, is_scale=is_scale, network_mode=network_mode)  
    occlusion_12, occlusion_21 = occlusion(flow_fw_12['full_res'], flow_bw_21['full_res'])
    occlusion_23, occlusion_32 = occlusion(flow_fw_23['full_res'], flow_bw_32['full_res'])    

    
    flow_fw_12_uint16 = flow_fw_12['full_res'] * 64. + 32768
    flow_fw_12_uint16 = tf.cast(flow_fw_12_uint16, tf.uint16)
    flow_bw_21_uint16 = flow_bw_21['full_res'] * 64. + 32768
    flow_bw_21_uint16 = tf.cast(flow_bw_21_uint16, tf.uint16)  
    flow_fw_23_uint16 = flow_fw_23['full_res'] * 64. + 32768
    flow_fw_23_uint16 = tf.cast(flow_fw_23_uint16, tf.uint16)  
    flow_bw_32_uint16 = flow_bw_32['full_res'] * 64. + 32768
    flow_bw_32_uint16 = tf.cast(flow_bw_32_uint16, tf.uint16)      
    
    flow_fw_12_color = flow_to_color(flow_fw_12['full_res'], mask=None, max_flow=256)
    flow_bw_21_color = flow_to_color(flow_bw_21['full_res'], mask=None, max_flow=256)
    flow_fw_23_color = flow_to_color(flow_fw_23['full_res'], mask=None, max_flow=256)
    flow_bw_32_color = flow_to_color(flow_bw_32['full_res'], mask=None, max_flow=256)    
    
    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)    
    saver = tf.train.Saver(var_list=restore_vars)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True     
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer()) 
    sess.run(iterator.initializer) 

    save_dir = '/'.join([dataset_name, 'sample', model_name])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)                
    saver.restore(sess, '%s/%s/model-%d' % (restore_model_dir, model_name, sample_step))                                                                            
    for i in range(data_num):   
        [np_flow_fw_12_color, np_flow_bw_21_color, np_flow_fw_23_color, np_flow_bw_32_color, np_flow_fw_12, np_flow_bw_21, np_flow_fw_23, np_flow_bw_32, 
        np_occlusion_12, np_occlusion_21, np_occlusion_23, np_occlusion_32] = sess.run(
            [flow_fw_12_color, flow_bw_21_color, flow_fw_23_color, flow_bw_32_color, flow_fw_12_uint16, flow_bw_21_uint16, flow_fw_23_uint16, flow_bw_32_uint16, 
             occlusion_12, occlusion_21, occlusion_23, occlusion_32])
        
        #h, w = np_flow_fw_12.shape[1:3]
        #ones_channel = tf.ones([1, h, w, 1])
        #flow_compare = np.zeros([2*h, 2*w, 3])
        #flow_compare[:h, :w, :] = np_flow_fw_12_color
        #flow_compare[h:2*h, :w, :] = np_flow_bw_21_color
        #flow_compare[:h, w:2*w, :] = np_flow_bw_32_color
        #flow_compare[h:2*h, w:2*w, :] = np_flow_fw_23_color
        
        #flow_compare = flow_compare * 255
        #flow_compare = flow_compare.astype('uint8')
        #misc.imsave('%s/flow_compare_%04d.png' % (save_dir, i), flow_compare)
        
        h, w = np_flow_fw_12.shape[1:3]
        ones_channel = np.ones([h, w, 1]) 
        np_flow_fw_12 = np_flow_fw_12[0]
        np_flow_bw_21 = np_flow_bw_21[0]
        np_flow_fw_23 = np_flow_fw_23[0]
        np_flow_bw_32 = np_flow_bw_32[0]
        np_flow_fw_12 = np.concatenate([np_flow_fw_12, ones_channel], -1)
        np_flow_bw_21 = np.concatenate([np_flow_bw_21, ones_channel], -1)
        np_flow_fw_23 = np.concatenate([np_flow_fw_23, ones_channel], -1)
        np_flow_bw_32 = np.concatenate([np_flow_bw_32, ones_channel], -1)
        np_flow_fw_12 = np_flow_fw_12.astype(np.uint16)
        np_flow_bw_21 = np_flow_bw_21.astype(np.uint16)
        np_flow_fw_23 = np_flow_fw_23.astype(np.uint16)
        np_flow_bw_32 = np_flow_bw_32.astype(np.uint16)    
        np_flow_fw_12 = rgb_bgr(np_flow_fw_12)
        np_flow_bw_21 = rgb_bgr(np_flow_bw_21)
        np_flow_fw_23 = rgb_bgr(np_flow_fw_23)
        np_flow_bw_32 = rgb_bgr(np_flow_bw_32)
        
        cv2.imwrite('%s/flow_fw_12_%04d.png' % (save_dir, i), np_flow_fw_12)
        cv2.imwrite('%s/flow_bw_21_%04d.png' % (save_dir, i), np_flow_bw_21)
        cv2.imwrite('%s/flow_fw_23_%04d.png' % (save_dir, i), np_flow_fw_23)
        cv2.imwrite('%s/flow_bw_32_%04d.png' % (save_dir, i), np_flow_bw_32)
        #misc.imsave('%s/flow_bw_21_%04d.png' % (save_dir, i), np_flow_bw_21_color[0])
        #misc.imsave('%s/flow_fw_23_%04d.png' % (save_dir, i), np_flow_fw_23_color[0])        
        #io.savemat('%s/flow_%04d.mat' % (save_dir, i), mdict={'flow_fw_12': np_flow_fw_12[0], 'flow_bw_21': np_flow_bw_21[0], 'flow_fw_23': np_flow_fw_23[0], 'flow_bw_32': np_flow_bw_32[0]})
        misc.imsave('%s/occlusion_12_%04d.png' %(save_dir, i), np_occlusion_12[0, :, :, 0])
        misc.imsave('%s/occlusion_21_%04d.png' %(save_dir, i), np_occlusion_21[0, :, :, 0])
        misc.imsave('%s/occlusion_23_%04d.png' %(save_dir, i), np_occlusion_23[0, :, :, 0])
        misc.imsave('%s/occlusion_32_%04d.png' %(save_dir, i), np_occlusion_32[0, :, :, 0])
        
        print('Finish %d/%d' % (i, data_num))      
        
    
def kitti_prediction(dataset_name, restore_model_dir, model_name, sample_step=100000, dataset_config={}, is_scale=True, num_parallel_calls=4, network_mode='v1'):
    dataset = KittiDataset(data_list_file=dataset_config['data_list_file'], img_dir=dataset_config['img_dir'])
    iterator = dataset.create_prediction_iterator(dataset.data_list, num_parallel_calls=num_parallel_calls)
    data_num = dataset.data_num
    batch_img0_raw, batch_img1_raw, batch_img2_raw, batch_img0, batch_img1, batch_img2 = iterator.get_next()
    
    flow_fw, flow_bw = pyramid_processing(batch_img0, batch_img1, batch_img2, train=False, trainable=False, is_scale=is_scale, network_mode=network_mode)
    img_shape = tf.shape(batch_img0)
    mask = tf.ones([1, img_shape[1], img_shape[2], 1])
    fb_err_img = flow_error_image(flow_fw['full_res'], -flow_bw['full_res'], mask_occ=mask)

    flow_fw_color = flow_to_color(flow_fw['full_res'], mask=None, max_flow=256)
    flow_bw_color = flow_to_color(flow_bw['full_res'], mask=None, max_flow=256)
    flow_bw_color_minus = flow_to_color(-flow_bw['full_res'], mask=None, max_flow=256)
    
    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)    
    
    saver = tf.train.Saver(var_list=restore_vars)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True     
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer()) 
    sess.run(iterator.initializer) 

    save_dir = '/'.join([dataset_name, 'sample', model_name])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)                
    saver.restore(sess, '%s/%s/model-%d' % (restore_model_dir, model_name, sample_step))        
    for i in range(data_num):   
        np_batch_img0, np_batch_img1, np_batch_img2, np_flow_fw_color, np_flow_bw_color, np_flow_fw, np_flow_bw, np_fb_err_img, np_flow_bw_color_minus = sess.run(
            [batch_img0_raw, batch_img1_raw, batch_img2_raw, flow_fw_color, flow_bw_color, flow_fw['full_res'], flow_bw['full_res'], fb_err_img, flow_bw_color_minus])
        
        h, w = np_batch_img1.shape[1:3]
        result_compare = np.zeros([3*h, 3*w, 3])
        result_compare[:h, :w, :] = (np_batch_img0 + np_batch_img1) / 2
        result_compare[h:2*h, :w, :] = np_flow_bw_color
        result_compare[2*h:3*h, :w, :] = np_fb_err_img
        result_compare[:h, w:2*w, :] = (np_batch_img1 + np_batch_img2) / 2
        result_compare[h:2*h, w:2*w, :] = np_flow_fw_color
        result_compare[2*h:3*h, w:2*w, :] = np_flow_bw_color_minus
        result_compare[:h, 2*w:3*w, :] = np_batch_img0
        result_compare[h:2*h, 2*w:3*w, :] = np_batch_img1
        result_compare[2*h:3*h, 2*w:3*w, :] = np_batch_img2

        result_compare = result_compare * 255
        result_compare = result_compare.astype('uint8')
        #isc.imsave('%s/result_%04d.jpg' % (save_dir, i), result_compare)
        misc.imsave('%s/flow_%04d.jpg' % (save_dir, i), np_flow_fw_color[0])
        #o.savemat('%s/result_%04d.mat' % (save_dir, i), mdict={'flow_fw': np_flow_fw})
        
        print('Finish %d/%d' % (i, data_num)) 
        

def sintel_prediction(dataset_name, restore_model_dir, model_name, sample_step=100000, dataset_config={}, is_scale=True, num_parallel_calls=4, network_mode='v1'):
    dataset = SintelDataset(data_list_file=dataset_config['data_list_file'], 
                            img_dir=dataset_config['img_dir'],
                            dataset_type=dataset_config['dataset_type'])
    iterator = dataset.create_prediction_iterator(dataset.data_list, num_parallel_calls=num_parallel_calls)
    data_num = dataset.data_num
    batch_img0_raw, batch_img1_raw, batch_img2_raw, batch_img0, batch_img1, batch_img2 = iterator.get_next()
    
    flow_fw, flow_bw = pyramid_processing(batch_img0, batch_img1, batch_img2, train=False, trainable=False, is_scale=is_scale, network_mode=network_mode)
    img_shape = tf.shape(batch_img0)
    mask = tf.ones([1, img_shape[1], img_shape[2], 1])
    fb_err_img = flow_error_image(flow_fw['full_res'], -flow_bw['full_res'], mask_occ=mask)

    flow_fw_color = flow_to_color(flow_fw['full_res'], mask=None, max_flow=256)
    flow_bw_color = flow_to_color(flow_bw['full_res'], mask=None, max_flow=256)
    flow_bw_color_minus = flow_to_color(-flow_bw['full_res'], mask=None, max_flow=256)
    
    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)    
    
    saver = tf.train.Saver(var_list=restore_vars)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True     
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer()) 
    sess.run(iterator.initializer) 

    save_dir = '/'.join([dataset_name, 'sample', model_name])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)                
    saver.restore(sess, '%s/%s/model-%d' % (restore_model_dir, model_name, sample_step))        
    for i in range(data_num):   
        np_batch_img0, np_batch_img1, np_batch_img2, np_flow_fw_color, np_flow_bw_color, np_flow_fw, np_flow_bw, np_fb_err_img, np_flow_bw_color_minus = sess.run(
            [batch_img0_raw, batch_img1_raw, batch_img2_raw, flow_fw_color, flow_bw_color, flow_fw['full_res'], flow_bw['full_res'], fb_err_img, flow_bw_color_minus])
        
        #h, w = np_batch_img1.shape[1:3]
        #result_compare = np.zeros([3*h, 3*w, 3])
        #result_compare[:h, :w, :] = (np_batch_img0 + np_batch_img1) / 2
        #result_compare[h:2*h, :w, :] = np_flow_bw_color
        #result_compare[2*h:3*h, :w, :] = np_fb_err_img
        #result_compare[:h, w:2*w, :] = (np_batch_img1 + np_batch_img2) / 2
        #result_compare[h:2*h, w:2*w, :] = np_flow_fw_color
        #result_compare[2*h:3*h, w:2*w, :] = np_flow_bw_color_minus
        #result_compare[:h, 2*w:3*w, :] = np_batch_img0
        #result_compare[h:2*h, 2*w:3*w, :] = np_batch_img1
        #result_compare[2*h:3*h, 2*w:3*w, :] = np_batch_img2

        #result_compare = result_compare * 255
        #result_compare = result_compare.astype('uint8')
        #misc.imsave('%s/result_%04d.jpg' % (save_dir, i), result_compare)
        misc.imsave('%s/flow_%04d.png' % (save_dir, i), np_flow_fw_color[0])
        #io.savemat('%s/result_%04d.mat' % (save_dir, i), mdict={'flow_fw': np_flow_fw[0]})
        
        print('Finish %d/%d' % (i, data_num)) 
        

def sintel_prediction_unsupervise(dataset_name, restore_model_dir, model_name, sample_step=100000, dataset_config={}, is_scale=True, num_parallel_calls=4, network_mode='v1'):
    dataset = SintelDataset(data_list_file=dataset_config['data_list_file'], img_dir=dataset_config['img_dir'])
    data_num = dataset.data_num
    iterator = dataset.create_prediction_unsupervise_iterator(dataset.data_list, num_parallel_calls=num_parallel_calls)
    batch_img0, batch_img1, batch_img2, batch_img3, batch_img4 = iterator.get_next()
    flow_fw_12, flow_bw_10, flow_fw_23, flow_bw_21, flow_fw_34, flow_bw_32 = pyramid_processing_five_frame(batch_img0, batch_img1, batch_img2, batch_img3, batch_img4, 
                                                                                                           train=False, trainable=False, regularizer=None, is_scale=is_scale, network_mode=network_mode)  
    occlusion_12, occlusion_21 = occlusion(flow_fw_12['full_res'], flow_bw_21['full_res'])
    occlusion_23, occlusion_32 = occlusion(flow_fw_23['full_res'], flow_bw_32['full_res'])    
    
    
    flow_fw_12_uint16 = flow_fw_12['full_res'] * 64. + 32768
    flow_fw_12_uint16 = tf.cast(flow_fw_12_uint16, tf.uint16)
    flow_bw_21_uint16 = flow_bw_21['full_res'] * 64. + 32768
    flow_bw_21_uint16 = tf.cast(flow_bw_21_uint16, tf.uint16)  
    flow_fw_23_uint16 = flow_fw_23['full_res'] * 64. + 32768
    flow_fw_23_uint16 = tf.cast(flow_fw_23_uint16, tf.uint16)  
    flow_bw_32_uint16 = flow_bw_32['full_res'] * 64. + 32768
    flow_bw_32_uint16 = tf.cast(flow_bw_32_uint16, tf.uint16)      
    
    flow_fw_12_color = flow_to_color(flow_fw_12['full_res'], mask=None, max_flow=128)
    flow_bw_21_color = flow_to_color(flow_bw_21['full_res'], mask=None, max_flow=128)
    flow_fw_23_color = flow_to_color(flow_fw_23['full_res'], mask=None, max_flow=128)
    flow_bw_32_color = flow_to_color(flow_bw_32['full_res'], mask=None, max_flow=128)    
    
    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)    
    saver = tf.train.Saver(var_list=restore_vars)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True     
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer()) 
    sess.run(iterator.initializer) 

    save_dir = '/'.join([dataset_name, 'sample', model_name])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)                
    saver.restore(sess, '%s/%s/model-%d' % (restore_model_dir, model_name, sample_step))                                                                            
    for i in range(data_num):   
        [np_flow_fw_12_color, np_flow_bw_21_color, np_flow_fw_23_color, np_flow_bw_32_color, np_flow_fw_12, np_flow_bw_21, np_flow_fw_23, np_flow_bw_32, 
        np_occlusion_12, np_occlusion_21, np_occlusion_23, np_occlusion_32] = sess.run(
            [flow_fw_12_color, flow_bw_21_color, flow_fw_23_color, flow_bw_32_color, flow_fw_12_uint16, flow_bw_21_uint16, flow_fw_23_uint16, flow_bw_32_uint16, 
             occlusion_12, occlusion_21, occlusion_23, occlusion_32])
        
        #h, w = np_flow_fw_12.shape[1:3]
        #flow_compare = np.zeros([2*h, 2*w, 3])
        #flow_compare[:h, :w, :] = np_flow_fw_12_color
        #flow_compare[h:2*h, :w, :] = np_flow_bw_21_color
        #flow_compare[:h, w:2*w, :] = np_flow_bw_32_color
        #flow_compare[h:2*h, w:2*w, :] = np_flow_fw_23_color
        
        #flow_compare = flow_compare * 255
        #flow_compare = flow_compare.astype('uint8')
        #misc.imsave('%s/flow_compare_%04d.png' % (save_dir, i), flow_compare)
        
        h, w = np_flow_fw_12.shape[1:3]
        ones_channel = np.ones([h, w, 1]) 
        np_flow_fw_12 = np_flow_fw_12[0]
        np_flow_bw_21 = np_flow_bw_21[0]
        np_flow_fw_23 = np_flow_fw_23[0]
        np_flow_bw_32 = np_flow_bw_32[0]
        np_flow_fw_12 = np.concatenate([np_flow_fw_12, ones_channel], -1)
        np_flow_bw_21 = np.concatenate([np_flow_bw_21, ones_channel], -1)
        np_flow_fw_23 = np.concatenate([np_flow_fw_23, ones_channel], -1)
        np_flow_bw_32 = np.concatenate([np_flow_bw_32, ones_channel], -1)
        np_flow_fw_12 = np_flow_fw_12.astype(np.uint16)
        np_flow_bw_21 = np_flow_bw_21.astype(np.uint16)
        np_flow_fw_23 = np_flow_fw_23.astype(np.uint16)
        np_flow_bw_32 = np_flow_bw_32.astype(np.uint16)    
        np_flow_fw_12 = rgb_bgr(np_flow_fw_12)
        np_flow_bw_21 = rgb_bgr(np_flow_bw_21)
        np_flow_fw_23 = rgb_bgr(np_flow_fw_23)
        np_flow_bw_32 = rgb_bgr(np_flow_bw_32)
        
        cv2.imwrite('%s/flow_fw_12_%04d.png' % (save_dir, i), np_flow_fw_12)
        cv2.imwrite('%s/flow_bw_21_%04d.png' % (save_dir, i), np_flow_bw_21)
        cv2.imwrite('%s/flow_fw_23_%04d.png' % (save_dir, i), np_flow_fw_23)
        cv2.imwrite('%s/flow_bw_32_%04d.png' % (save_dir, i), np_flow_bw_32)
        #misc.imsave('%s/flow_bw_21_%04d.png' % (save_dir, i), np_flow_bw_21_color[0])
        #misc.imsave('%s/flow_fw_23_%04d.png' % (save_dir, i), np_flow_fw_23_color[0])
        #io.savemat('%s/flow_%04d.mat' % (save_dir, i), mdict={'flow_fw_12': np_flow_fw_12[0], 'flow_bw_21': np_flow_bw_21[0], 'flow_fw_23': np_flow_fw_23[0], 'flow_bw_32': np_flow_bw_32[0]})
        misc.imsave('%s/occlusion12_%04d.png' %(save_dir, i), np_occlusion_12[0, :, :, 0])
        misc.imsave('%s/occlusion21_%04d.png' %(save_dir, i), np_occlusion_21[0, :, :, 0])
        misc.imsave('%s/occlusion23_%04d.png' %(save_dir, i), np_occlusion_23[0, :, :, 0])
        misc.imsave('%s/occlusion32_%04d.png' %(save_dir, i), np_occlusion_32[0, :, :, 0])        
        print('Finish %d/%d' % (i, data_num))  

def sintel_raw_prediction(dataset_name, restore_model_dir, model_name, sample_step=100000, dataset_config={}, is_scale=True, num_parallel_calls=4, network_mode='v1'):
    dataset = SintelRawDataset(data_list_file=dataset_config['data_list_file'], img_dir=dataset_config['img_dir'])
    data_num = dataset.data_num
    iterator = dataset.create_prediction_iterator(dataset.data_list, num_parallel_calls=num_parallel_calls)
    batch_img0, batch_img1, batch_img2, batch_img3, batch_img4 = iterator.get_next()
    flow_fw_12, flow_bw_10, flow_fw_23, flow_bw_21, flow_fw_34, flow_bw_32 = pyramid_processing_five_frame(batch_img0, batch_img1, batch_img2, batch_img3, batch_img4, 
                                                                                                           train=False, trainable=False, regularizer=None, is_scale=is_scale, network_mode=network_mode)  
    occlusion_12, occlusion_21 = occlusion(flow_fw_12['full_res'], flow_bw_21['full_res'])
    occlusion_23, occlusion_32 = occlusion(flow_fw_23['full_res'], flow_bw_32['full_res'])    
    
    mag_sq = length_sq(flow_bw_21['full_res']) + length_sq(flow_fw_23['full_res'])
    occ_thresh =  0.01 * mag_sq + 0.5
    occlusion_2 = tf.cast(length_sq(flow_bw_21['full_res'] + flow_fw_23['full_res']) > occ_thresh, tf.float32)
    
    flow_fw_12_color = flow_to_color(flow_fw_12['full_res'], mask=None, max_flow=256)
    flow_bw_21_color = flow_to_color(flow_bw_21['full_res'], mask=None, max_flow=256)
    flow_fw_23_color = flow_to_color(flow_fw_23['full_res'], mask=None, max_flow=256)
    flow_bw_32_color = flow_to_color(flow_bw_32['full_res'], mask=None, max_flow=256)    
    
    flow_fw_12_uint16 = flow_fw_12['full_res'] * 64. + 32768
    flow_fw_12_uint16 = tf.cast(flow_fw_12_uint16, tf.uint16)
    flow_bw_21_uint16 = flow_bw_21['full_res'] * 64. + 32768
    flow_bw_21_uint16 = tf.cast(flow_bw_21_uint16, tf.uint16)  
    flow_fw_23_uint16 = flow_fw_23['full_res'] * 64. + 32768
    flow_fw_23_uint16 = tf.cast(flow_fw_23_uint16, tf.uint16)  
    flow_bw_32_uint16 = flow_bw_32['full_res'] * 64. + 32768
    flow_bw_32_uint16 = tf.cast(flow_bw_32_uint16, tf.uint16)     
    
    
    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)    
    saver = tf.train.Saver(var_list=restore_vars)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True     
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer()) 
    sess.run(iterator.initializer) 

    save_dir = '/'.join([dataset_name, 'sample', model_name])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)                
    saver.restore(sess, '%s/%s/model-%d' % (restore_model_dir, model_name, sample_step))                                                                            
    for i in range(data_num):   
        [np_flow_fw_12_color, np_flow_bw_21_color, np_flow_fw_23_color, np_flow_bw_32_color, np_flow_fw_12, np_flow_bw_21, np_flow_fw_23, np_flow_bw_32, 
        np_occlusion_12, np_occlusion_21, np_occlusion_23, np_occlusion_32, np_occlusion_2] = sess.run(
            [flow_fw_12_color, flow_bw_21_color, flow_fw_23_color, flow_bw_32_color, flow_fw_12_uint16, flow_bw_21_uint16, flow_fw_23_uint16, flow_bw_32_uint16, 
             occlusion_12, occlusion_21, occlusion_23, occlusion_32, occlusion_2])
        
        #h, w = np_flow_fw_12.shape[1:3]
        #flow_compare = np.zeros([2*h, 2*w, 3])
        #flow_compare[:h, :w, :] = np_flow_fw_12_color
        #flow_compare[h:2*h, :w, :] = np_flow_bw_21_color
        #flow_compare[:h, w:2*w, :] = np_flow_bw_32_color
        #flow_compare[h:2*h, w:2*w, :] = np_flow_fw_23_color
        
        #flow_compare = flow_compare * 255
        #flow_compare = flow_compare.astype('uint8')
        #misc.imsave('%s/flow_compare_%05d.png' % (save_dir, i), flow_compare)
        h, w = np_flow_fw_12.shape[1:3]
        ones_channel = np.ones([h, w, 1]) 
        np_flow_fw_12 = np_flow_fw_12[0]
        np_flow_bw_21 = np_flow_bw_21[0]
        np_flow_fw_23 = np_flow_fw_23[0]
        np_flow_bw_32 = np_flow_bw_32[0]
        np_flow_fw_12 = np.concatenate([np_flow_fw_12, ones_channel], -1)
        np_flow_bw_21 = np.concatenate([np_flow_bw_21, ones_channel], -1)
        np_flow_fw_23 = np.concatenate([np_flow_fw_23, ones_channel], -1)
        np_flow_bw_32 = np.concatenate([np_flow_bw_32, ones_channel], -1)
        np_flow_fw_12 = np_flow_fw_12.astype(np.uint16)
        np_flow_bw_21 = np_flow_bw_21.astype(np.uint16)
        np_flow_fw_23 = np_flow_fw_23.astype(np.uint16)
        np_flow_bw_32 = np_flow_bw_32.astype(np.uint16)    
        np_flow_fw_12 = rgb_bgr(np_flow_fw_12)
        np_flow_bw_21 = rgb_bgr(np_flow_bw_21)
        np_flow_fw_23 = rgb_bgr(np_flow_fw_23)
        np_flow_bw_32 = rgb_bgr(np_flow_bw_32)
        
        cv2.imwrite('%s/flow_fw_12_%05d.png' % (save_dir, i), np_flow_fw_12)
        cv2.imwrite('%s/flow_bw_21_%05d.png' % (save_dir, i), np_flow_bw_21)
        cv2.imwrite('%s/flow_fw_23_%05d.png' % (save_dir, i), np_flow_fw_23)
        cv2.imwrite('%s/flow_bw_32_%05d.png' % (save_dir, i), np_flow_bw_32)
        
        #io.savemat('%s/flow_%04d.mat' % (save_dir, i), mdict={'flow_fw_12': np_flow_fw_12[0], 'flow_bw_21': np_flow_bw_21[0], 'flow_fw_23': np_flow_fw_23[0], 'flow_bw_32': np_flow_bw_32[0]})
        misc.imsave('%s/occlusion12_%05d.png' %(save_dir, i), np_occlusion_12[0, :, :, 0])
        misc.imsave('%s/occlusion21_%05d.png' %(save_dir, i), np_occlusion_21[0, :, :, 0])
        misc.imsave('%s/occlusion23_%05d.png' %(save_dir, i), np_occlusion_23[0, :, :, 0])
        misc.imsave('%s/occlusion32_%05d.png' %(save_dir, i), np_occlusion_32[0, :, :, 0])
        #misc.imsave('%s/occlusion2_%05d.png' %(save_dir, i), np_occlusion_2[0, :, :, 0])
        
        print('Finish %d/%d' % (i, data_num))  

def kitti_sample(dataset_name, restore_model_dir, model_name, sample_step=100000, dataset_config={}, is_scale=True, num_parallel_calls=4, network_mode='v1'):
    dataset, _ = create_dataset_and_iterator(dataset_name, dataset_config, num_parallel_calls)
    iterator = dataset.create_sample_2015_iterator(dataset.data_list, num_parallel_calls=num_parallel_calls)
    data_num = dataset.data_num
    batch_img0, batch_img1, batch_img2, batch_img3, batch_img4, batch_flow, batch_mask = iterator.get_next()
    
    flow_fw_12, flow_bw_10, flow_fw_23, flow_bw_21, flow_fw_34, flow_bw_32 = pyramid_processing_five_frame(
        batch_img0, batch_img1, batch_img2, batch_img3, batch_img4, train=False, trainable=False, is_scale=is_scale, network_mode=network_mode)
    img_shape = tf.shape(batch_img0)
    mask = tf.ones([1, img_shape[1], img_shape[2], 1])
    flow_fw_err_img = flow_error_image(flow_fw_23['full_res'], batch_flow, mask_occ=batch_mask)
    flow_bw_err_img = flow_error_image(-flow_bw_21['full_res'], batch_flow, mask_occ=batch_mask)
    occ_fw, _ = occlusion(flow_fw_23['full_res'], flow_bw_32['full_res'])
    _, occ_bw = occlusion(flow_fw_12['full_res'], flow_bw_21['full_res'])
    occ_fw_sparse = tf.multiply(occ_fw, batch_mask)
    occ_bw_sparse = tf.multiply(occ_bw, batch_mask)
    occ_f_noc_b = tf.clip_by_value(occ_fw_sparse - occ_bw_sparse, 0., 1.)

    flow_fw_color = flow_to_color(flow_fw_23['full_res'], mask=None, max_flow=256)
    flow_bw_color = flow_to_color(flow_bw_32['full_res'], mask=None, max_flow=256)
    flow_bw_color_minus = flow_to_color(-flow_bw_32['full_res'], mask=None, max_flow=256)
    
    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)    
    
    saver = tf.train.Saver(var_list=restore_vars)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True     
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer()) 
    sess.run(iterator.initializer) 

    save_dir = '/'.join([dataset_name, 'sample', model_name])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)                
    saver.restore(sess, '%s/%s/model-%d' % (restore_model_dir, model_name, sample_step))        
    for i in range(data_num):   
        np_flow_fw_err_img, np_flow_bw_err_img, np_occ_fw, np_occ_bw, np_occ_fw_sparse, np_occ_bw_sparse, np_occ_f_noc_b = sess.run(
            [flow_fw_err_img, flow_bw_err_img, occ_fw, occ_bw, occ_fw_sparse, occ_bw_sparse, occ_f_noc_b])
        np_occ_fw = np.repeat(np_occ_fw, 3, -1)
        np_occ_bw = np.repeat(np_occ_bw, 3, -1)
        np_occ_fw_sparse = np.repeat(np_occ_fw_sparse, 3, -1)
        np_occ_bw_sparse = np.repeat(np_occ_bw_sparse, 3, -1)
        np_occ_f_noc_b = np.repeat(np_occ_f_noc_b, 3, -1)
        
        
        h, w = np_flow_fw_err_img.shape[1:3]
        result_compare = np.zeros([3*h, 2*w, 3])
        result_compare[:h, :w, :] = np_occ_f_noc_b[0] * np_flow_fw_err_img[0]
        result_compare[h:2*h, :w, :] = np_flow_fw_err_img[0]
        result_compare[2*h:3*h, :w, :] = np_occ_fw_sparse[0]
        result_compare[:h, w:2*w, :] = np_occ_f_noc_b[0] * np_flow_bw_err_img[0]
        result_compare[h:2*h, w:2*w, :] = np_flow_bw_err_img[0]
        result_compare[2*h:3*h, w:2*w, :] = np_occ_bw_sparse[0]

        result_compare = result_compare * 255
        result_compare = result_compare.astype('uint8')
        misc.imsave('%s/result_%d.png' % (save_dir, i), result_compare)
        #io.savemat('%s/result_%d.mat' % (save_dir, i), mdict={'flow_fw': np_flow_fw, 'flow_bw': np_flow_bw})
        
        print('Finish %d/%d' % (i, data_num)) 

def prediction(dataset_name, restore_model_dir, model_name, sample_step=100000, dataset_config={}, is_scale=True, num_parallel_calls=4, network_mode='v1'):     
    if dataset_name == 'Kitti':
        kitti_prediction(dataset_name, restore_model_dir, model_name, sample_step, dataset_config, is_scale, num_parallel_calls, network_mode=network_mode)
        #prediction_new(dataset_name, restore_model_dir, model_name, sample_step, dataset_config, is_scale, num_parallel_calls, network_mode=network_mode)
        #kitti_sample(dataset_name, restore_model_dir, model_name, sample_step, dataset_config, is_scale, num_parallel_calls, network_mode=network_mode)
        #sample(dataset_name, restore_model_dir, model_name, sample_step, dataset_config, is_scale, num_parallel_calls, network_mode=network_mode)
    elif dataset_name == 'KittiRaw':
        kitti_raw_prediction(dataset_name, restore_model_dir, model_name, sample_step, dataset_config, is_scale, num_parallel_calls, network_mode=network_mode)
    elif dataset_name == 'Sintel':
        #sintel_prediction(dataset_name, restore_model_dir, model_name, sample_step, dataset_config, is_scale, num_parallel_calls, network_mode=network_mode) 
        sintel_prediction_unsupervise(dataset_name, restore_model_dir, model_name, sample_step, dataset_config, is_scale, num_parallel_calls, network_mode=network_mode) 
        #sample(dataset_name, restore_model_dir, model_name, sample_step, dataset_config, is_scale, num_parallel_calls, network_mode=network_mode)
    elif dataset_name == 'SintelRaw':
        sintel_raw_prediction(dataset_name, restore_model_dir, model_name, sample_step, dataset_config, is_scale, num_parallel_calls, network_mode=network_mode)     
    else:
        raise ValueError('Invalid dataset. Dataset should be one of {Kitti, KittRaw, Sintel}')    

        

def distillation_prediction(dataset_name, restore_model_dir, model_name, sample_step=100000, dataset_config={}, is_scale=True, num_parallel_calls=4):
    dataset, _ = create_dataset_and_iterator(dataset_name, dataset_config, num_parallel_calls)
    iterator = dataset.create_prediction_iterator(dataset.data_list, num_parallel_calls=num_parallel_calls)
    data_num = dataset.data_num
    batch_img1_raw, batch_img2_raw, batch_img1, batch_img2 = iterator.get_next()
    
    flow_estimated = pyramid_processing(batch_img1, batch_img2, train=False, trainable=False, is_scale=is_scale)

    flow_est_color = flow_to_color(flow_estimated['full_res'], mask=None, max_flow=128)
    
    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)    
    
    saver = tf.train.Saver(var_list=restore_vars)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True     
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer()) 
    sess.run(iterator.initializer) 

    save_dir = '/'.join([dataset_name, 'sample', model_name])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)                
    saver.restore(sess, '%s/%s/model-%d' % (restore_model_dir, model_name, sample_step)) 
    #for i in range(data_num):
        #sess.run([batch_img1_raw, batch_img2_raw, flow_est_color, flow_estimated['full_res']])        
    for i in range(data_num):   
        np_batch_img1, np_batch_img2, np_flow_est_color, np_flow_estimated = sess.run(
            [batch_img1_raw, batch_img2_raw, flow_est_color, flow_estimated['full_res']])
        
        h, w = np_batch_img1.shape[1:3]
        result_compare = np.zeros([2*h, w, 3])
        result_compare[:h, :, :] = np_batch_img1
        #result_compare[h:2*h, :, :] = np_batch_img2
        result_compare[h:2*h, :, :] = np_flow_est_color
        #result_compare[3*h:, :, :] = (np_batch_img1 + np_batch_img2) / 2

        result_compare = result_compare * 255
        result_compare = result_compare.astype('uint8')
        #misc.imsave('%s/result_%d.png' % (save_dir, i), result_compare)
        io.savemat('%s/result_%d.mat' % (save_dir, i), mdict={'flow_est': np_flow_estimated})
        
        print('Finish %d/%d' % (i, data_num))  
    
def prediction_new(dataset_name, restore_model_dir, model_name, sample_step=100000, dataset_config={}, is_scale=True, num_parallel_calls=4, network_mode='v1'):
    dataset = KittiDataset(data_list_file=dataset_config['data_list_file'], img_dir=dataset_config['img_dir'])
    iterator = dataset.create_prediction_iterator(dataset.data_list, num_parallel_calls=num_parallel_calls)
    data_num = dataset.data_num
    batch_img0_raw, batch_img1_raw, batch_img2_raw, batch_img0, batch_img1, batch_img2 = iterator.get_next()
    
    flow_estimated, _ = pyramid_processing(batch_img0, batch_img1, batch_img2, train=False, trainable=False, is_scale=is_scale, network_mode=network_mode)

    flow_est_color = flow_to_color(flow_estimated['full_res'], mask=None, max_flow=128)
    
    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)    
    
    saver = tf.train.Saver(var_list=restore_vars)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) 
    sess.run(iterator.initializer) 

    save_dir = '/'.join([dataset_name, 'sample', model_name])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)                
    saver.restore(sess, '%s/%s/model-%d' % (restore_model_dir, model_name, sample_step)) 
    for i in range(data_num):   
        np_batch_img1, np_batch_img2, np_flow_est_color, np_flow_estimated = sess.run(
            [batch_img1_raw, batch_img2_raw, flow_est_color, flow_estimated['full_res']])
        
        #h, w = np_batch_img1.shape[1:3]
        #result_compare = np.zeros([2*h, w, 3])
        #result_compare[:h, :, :] = np_batch_img1
        #result_compare[h:2*h, :, :] = np_batch_img2
        #result_compare[h:2*h, :, :] = np_flow_est_color
        #result_compare[3*h:, :, :] = (np_batch_img1 + np_batch_img2) / 2

        #result_compare = result_compare * 255
        #result_compare = result_compare.astype('uint8')
        #misc.imsave('%s/result_%d.jpg' % (save_dir, i), result_compare)
        misc.imsave('%s/flow_%04d.png' % (save_dir, i), np_flow_est_color[0])
        #io.savemat('%s/result_%d.mat' % (save_dir, i), mdict={'flow_est': np_flow_estimated})
        
        print('Finish %d/%d' % (i, data_num))     


def sample(dataset_name, restore_model_dir, model_name, sample_step=100000, dataset_config={}, is_scale=True, num_parallel_calls=4, network_mode='v1'):
    dataset, iterator = create_dataset_and_iterator(dataset_name, dataset_config, num_parallel_calls)
    data_num = dataset.data_num
    batch_img0_raw, batch_img1_raw, batch_img2_raw, batch_img0, batch_img1, batch_img2, batch_flow, batch_mask = generate_batch_input(dataset_name, iterator)
    
    flow_estimated, _ = pyramid_processing(batch_img0, batch_img1, batch_img2, train=False, trainable=False, is_scale=is_scale, network_mode=network_mode)
    
    flow_gt_color = flow_to_color(batch_flow, mask=batch_mask, max_flow=128)
    flow_est_color = flow_to_color(flow_estimated['full_res'], mask=batch_mask, max_flow=128)
    flow_est_color_all = flow_to_color(flow_estimated['full_res'], mask=None, max_flow=128)
    flow_err_color = flow_error_image(batch_flow, flow_estimated['full_res'], mask_occ=tf.ones_like(batch_mask))
    
    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)    
    
    saver = tf.train.Saver(var_list=restore_vars)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) 
    sess.run(iterator.initializer) 

    save_dir = '/'.join([dataset_name, 'sample', model_name])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)                
    saver.restore(sess, '%s/%s/model-%d' % (restore_model_dir, model_name, sample_step)) 
    for i in range(data_num):   
        np_batch_img1, np_batch_img2, np_flow_gt_color, np_flow_est_color, np_flow_est_color_all, np_flow_err_color = sess.run(
            [batch_img1_raw, batch_img2_raw, flow_gt_color, flow_est_color, flow_est_color_all, flow_err_color])
        
        h, w = np_batch_img1.shape[1:3]
        np_batch_img1 = np_batch_img1[0]
        np_batch_img2 = np_batch_img2[0]
        np_flow_gt_color = np_flow_gt_color[0]
        np_flow_est_color = np_flow_est_color[0]
        np_flow_est_color_all = np_flow_est_color_all[0]
        np_flow_err_color = np_flow_err_color[0]
        
        #result_compare = np.zeros([4*h, w, 3])
        #result_compare[:h, :w, :] = np_batch_img1
        #result_compare[h:2*h, :w, :] = np_flow_gt_color
        #result_compare[2*h:3*h, :w, :] = np_flow_est_color_all
        #result_compare[3*h:4*h, :w, :] = np_flow_err_color
                     
        #result_compare = result_compare * 255
        #result_compare = result_compare.astype('uint8')
        #misc.imsave('%s/result_%04d.png' % (save_dir, i), result_compare)
        misc.imsave('%s/img1_%04d.jpg' %(save_dir, i), np_batch_img1)
        misc.imsave('%s/flow_gt_%04d.jpg' %(save_dir, i), np_flow_gt_color)
        misc.imsave('%s/flow_est_%04d.jpg' %(save_dir, i), np_flow_est_color_all)
        misc.imsave('%s/flow_err_%04d.jpg' %(save_dir, i), np_flow_err_color)
        print('Finish %d/%d' % (i, data_num))


def main(_):
    config = config_dict('./config/eval_config.ini')
    run_config = config['run']
    dataset_config = config[run_config['dataset']] 
    mode_config = config[run_config['mode']]
    if run_config['mode'] == 'test':
        test(dataset_name=run_config['dataset'],
             restore_model_dir=run_config['restore_model_dir'],
             model_name=dataset_config['model_name'], 
             start_step=mode_config['start_step'],
             end_step=mode_config['end_step'],
             checkpoint_interval=mode_config['checkpoint_interval'],
             dataset_config=dataset_config,
             is_scale=run_config['is_scale'],
             is_write_summmary=mode_config['is_write_summmary'],
             num_parallel_calls=run_config['num_parallel_calls'],
             network_mode=run_config['network_mode'])
    elif run_config['mode'] == 'predict':
        prediction(dataset_name=run_config['dataset'],
                   restore_model_dir=run_config['restore_model_dir'],
                   model_name=dataset_config['model_name'], 
                   sample_step=mode_config['sample_step'],  
                   dataset_config=dataset_config, 
                   is_scale=run_config['is_scale'], 
                   num_parallel_calls=run_config['num_parallel_calls'],
                   network_mode=run_config['network_mode'])
    else:
        raise ValueError('Invalid mode. Mode should be one of {test, sample, predict}')


if __name__ == '__main__':
    tf.app.run() 
