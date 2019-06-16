# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import cv2
import matplotlib.pyplot as plt
from flowlib import read_flo, read_pfm
from data_augmentation import *
from utils import mvn   

class BasicDataset(object):
    def __init__(self, crop_h=320, crop_w=896, batch_size=4, data_list_file='path_to_your_data_list_file', 
                 img_dir='path_to_your_image_directory', fake_flow_occ_dir='path_to_your_fake_flow_occlusion_directory'):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.batch_size = batch_size
        self.img_dir = img_dir
        self.data_list = np.loadtxt(data_list_file, dtype=np.str)
        self.data_num = self.data_list.shape[0]
        self.fake_flow_occ_dir = fake_flow_occ_dir
    
    # KITTI's data format for storing flow and mask
    # The first two channels are flow, the third channel is mask
    def extract_flow_and_mask(self, flow):
        optical_flow = flow[:, :, :2]
        optical_flow = (optical_flow - 32768) / 64.0
        mask = tf.cast(tf.greater(flow[:, :, 2], 0), tf.float32)
        #mask = tf.cast(flow[:, :, 2], tf.float32)
        mask = tf.expand_dims(mask, -1)
        return optical_flow, mask    
    
    # The default image type is PNG.
    def read_and_decode(self, filename_queue):
        img0_name = tf.string_join([self.img_dir, filename_queue[0]])
        img1_name = tf.string_join([self.img_dir, filename_queue[1]])
        img2_name = tf.string_join([self.img_dir, filename_queue[2]])
 
        img0 = tf.image.decode_png(tf.read_file(img0_name), channels=3)
        img0 = tf.cast(img0, tf.float32)        
        img1 = tf.image.decode_png(tf.read_file(img1_name), channels=3)
        img1 = tf.cast(img1, tf.float32)
        img2 = tf.image.decode_png(tf.read_file(img2_name), channels=3)
        img2 = tf.cast(img2, tf.float32)    
 
        return img0, img1, img2    
    

    def preprocess_one_shot(self, filename_queue):
        img0, img1, img2 = self.read_and_decode(filename_queue)
        img0 = img0 / 255.
        img1 = img1 / 255.
        img2 = img2 / 255.
        
        # normalize img
        #img0 = mvn(img0)
        #img1 = mvn(img1)
        #img2 = mvn(img2)        
        return img0, img1, img2

    
    def create_one_shot_iterator(self, data_list, num_parallel_calls=4):
        """ For Validation or Testing
            Generate image and flow one_by_one without cropping, image and flow size may change every iteration
        """
        data_list = tf.convert_to_tensor(data_list, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(data_list)
        dataset = dataset.map(self.preprocess_one_shot, num_parallel_calls=num_parallel_calls)        
        dataset = dataset.batch(1)
        dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        return iterator     