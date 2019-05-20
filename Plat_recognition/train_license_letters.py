import tensorflow as tf
import numpy as np
import sys
import time
import os
import random
from PIL import Image

SIZE = 1280
WIDTH = 32
HEIGHT = 40
NUM_CLASSES = 26
iterations = 300

SAVER_DIR = 'E:/tensorflow_model/plat_recognition/train_saver/letters/'
LETTERS_DIGITS = ("A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z","I","O")
license_num = ""
time_begin = time.time()

x = tf.placeholder(tf.float32,shape=[None,SIZE])
y = tf.placeholder(tf.float32,shape=[None,NUM_CLASSES])

x_image = tf.reshape(x,[-1,WIDTH,HEIGHT,1])

# 定义卷积层
def conv_layer(input,W,b,conv_stride,kernel_size,pool_stride,padding):
    L1_conv = tf.nn.conv2d(input,W,strides=conv_stride,padding=padding)
    L1_relu = tf.nn.relu(L1_conv+b)
    return tf.nn.max_pool(L1_relu,ksize=kernel_size,strides=pool_stride,padding='SAME')

# 定义全连接层,返回池化后的结果
def full_connect(input,W,b):
    return tf.nn.relu(tf.matmul(input,W)+b)