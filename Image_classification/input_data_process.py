import tensorflow as tf
import numpy as np
import os
from train import *

def get_files(train_dir):
    input_count = 0
    directors = []
    labels = []
    for filename in os.listdir(train_dir):
        path = os.path.join(train_dir,filename)
        if os.path.isdir(path):
            directors.append(path)
            labels.append(filename)

    image_list = []
    label_list = []
    i = 0
    for directory in directors:
        for filename in os.listdir(directory):
            path = os.path.join(directory,filename)
            image_list.append(path)
            label_list.append(i)
            input_count += 1
        i += 1

    # 打乱文件顺序
    temp = np.array([image_list,label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]
    return image_list,label_list,input_count

def get_batch(image,label,image_W,image_H,batch_size,capacity):
    # image, label: 要生成batch的图像和标签list
    # image_W, image_H: 图片的宽高
    # batch_size: 每个batch有多少张图片
    # capacity: 队列容量
    # return: 图像和标签的batch

    # 将python.list类型转换tensorflow能够识别的格式
    image = tf.cast(image,tf.string) #把文件路径转化为字符串格式存在list中
    label = tf.cast(label,tf.int32)

    # 生成队列
    input_queue = tf.train.slice_input_producer([image,label])
    # tf.image.decode_jepg和tf.image.decode_png分别用于解码jpg格式和png格式的图像进行解码，得到图像的像素值，这个像素值可以用于显示图像。
    # 如果没有解码，读取的图像是一个字符串，没法显示
    image_contents = tf.read_file(input_queue[0])
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents,channels=3)

    image = tf.image.resize_images(image,[image_H,image_W],method = tf.image.ResizeMethod.NEAREST_NEIGHBOR) #采用NEAREST_NEIGHBOR插值方法
    image = tf.cast(image,tf.float32)
    image_batch,label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=64,  # 线程
        capacity=capacity
    )
    #最后将得到的image_batch和label_batch返回。image_batch是一个4D的tensor，[batch, width, height, channels]，label_batch是一个1D的tensor，[batch]
    return image_batch,label_batch









