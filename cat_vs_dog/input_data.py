import tensorflow as tf
import numpy as np
import os

#获取文件路径和标签（预处理数据集，建立图片和标签之间的对应关系）
def get_files(file_dir):
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []

    for file in os.listdir(file_dir):
        name = file.split(sep = '.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0) #猫的标签设为0
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1) #狗的标签设为1
    print("There are %d cats \nThere are %d dogs" % (len(cats),len(dogs)))

    #打乱文件顺序
    image_list = np.hstack((cats,dogs)) #np.hstack:将两个向量连接起来成为一个向量
    label_list = np.hstack((label_cats,label_dogs))
    temp = np.array([image_list,label_list]) #每张图片的路径加标签
    temp = temp.transpose() #转置成[None,2]
    np.random.shuffle(temp) #打乱顺序

    image_list = list(temp[:,0]) #取出第一列（数据集中图片的路径）
    label_list = list(temp[:,1]) #去除第二列（数据集中图片的标签）
    label_list = [int(i) for i in label_list]
    return image_list,label_list #返回每张图片的路径列表和标签列表

#生成相同大小的批次
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


















