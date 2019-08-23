import tensorflow as tf 
import os
import numpy as np
from PIL import Image
import time
import math
from model import VGGNet

vgg_16_npy_pyth = './vgg16.npy'
content_img_path = './timg.jpg'
style_img_path = './nahan.jpg'
output_dir = './transfer/'

num_steps = 1000
learning_rate = 10

lambda_c = 1
lambda_s = 1000

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def initial_result(shape, mean, stddev):
    '''
    定义一个初始化好的随机图片，然后在该图片上不停的梯度下降来得到效果。
    :param shape: 输入形状
    :param mean: 均值
    :param stddev: 方法
    :return: 图片
    '''   
    initial = tf.truncated_normal(shape, mean = mean, stddev = stddev)
    return tf.Variable(initial)

def read_img(img_path):
    '''
    读取图片
    :param img_name: 图片路径
    :return: 4维矩阵
    '''
    img = Image.open(img_path)
    # 图像为三通道（224， 244， 3），但是需要转化为4维
    np_img = np.array(img)
    np_img = np.asarray([np_img], dtype = np.int32)
    return np_img

def gram_matrix(x):
    '''
    计算 gram 矩阵
    :param x: 特征图，shape：[1, width, height, channel]
    :return:
    ''' 
    b, w, h, ch = x.get_shape().as_list()
    # 这里求出来的是 每一个feature map之间的相似度
    features = tf.reshape(x, [b, w * h, ch])
    # 相似度矩阵 方法： 将矩阵转置为[ch, h*w], 再乘原矩阵，最后的矩阵是[ch , ch]
    # 防止矩阵数值过大，除以一个常数
    # 参数3， 表示将第一个参数转置
    gram = tf.matmul(features, features, adjoint_a = True) / tf.constant(ch * w * h, tf.float32)
    return gram

def main():
    # 生成一个图像,均值为127.5，方差为20
    result = initial_result((1,466,712,3), 127.5, 20)

    # 读取内容图像和风格图像
    content_val = read_img(content_img_path)
    style_val = read_img(style_img_path)

    content = tf.placeholder(tf.float32, shape = [1, 466, 712, 3])
    style = tf.placeholder(tf.float32, shape = [1, 615, 500, 3])

    # 载入模型，注意：在python3中，需要添加一句： encoding='latin1'
    data_dict = np.load(vgg_16_npy_pyth, encoding = 'latin1').item()

    # 创建这三张图像的 vgg 对象
    vgg_for_content = VGGNet(data_dict)
    vgg_for_style = VGGNet(data_dict)
    vgg_for_result = VGGNet(data_dict)

    # 创建每个神经网络
    vgg_for_content.build(content)
    vgg_for_style.build(style)
    vgg_for_result.build(result)

    # 提取哪些层特征
    # 需要注意的是：内容特征抽取的层数和结果特征抽取的层数必须相同
    # 风格特征抽取的层数和结果特征抽取的层数必须相同
    content_features = [
        # vgg_for_content.conv1_2,
        # vgg_for_content.conv2_2,
        # vgg_for_content.conv3_3,
        vgg_for_content.conv4_3,
        vgg_for_content.conv5_3,
    ]

    result_content_features = [
        # vgg_for_result.conv1_2,
        # vgg_for_result.conv2_2,
        # vgg_for_result.conv3_3,
        vgg_for_result.conv4_3,
        vgg_for_result.conv5_3,
    ]

    style_features = [
        vgg_for_style.conv2_2,
    ]

    result_style_features = [
        vgg_for_result.conv2_2,
    ]

    style_gram = [gram_matrix(feature) for feature in style_features]
    result_style_gram = [gram_matrix(feature) for feature in result_style_features]

    # 计算内容损失
    content_loss = tf.zeros(1, tf.float32)
    for c, c_ in zip(content_features, result_content_features):
        content_loss += tf.reduce_mean((c - c_) ** 2, axis = [1, 2, 3])
    
    # 计算风格损失
    style_loss = tf.zeros(1, tf.float32)
    for s, s_ in zip(style_gram, result_style_gram):
        # 因为在计算gram矩阵的时候，降低了一维，所以，只需要在[1, 2]两个维度求均值即可
        style_loss += tf.reduce_mean((s - s_) ** 2, axis = [1, 2])
    
    # 总的损失函数
    loss = content_loss * lambda_c + style_loss * lambda_s

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("1111111")
        for step in range(num_steps):
            loss_value, content_loss_value, style_loss_value, _ = sess.run([loss, content_loss, style_loss, train_op],
                feed_dict = {content:content_val, style:style_val}
            )

            print('step: %d, loss_value: %.4f, content_loss: %.4f, style_loss: %.4f' %
                (step+1, loss_value[0], content_loss_value[0], style_loss_value[0])
            )   
            if step % 100 == 0:
                result_img_path = os.path.join(output_dir, 'result_%05d.jpg' % (step+1))
                # 将图像取出，因为之前是4维，所以需要使用一个索引0，将其取出
                result_val = result.eval(sess)[0]
                # np.clip() numpy.clip(a, a_min, a_max, out=None)[source]
                # 其中a是一个数组，后面两个参数分别表示最小和最大值
                result_val = np.clip(result_val, 0, 255)

                img_arr = np.asarray(result_val, np.uint8)
                img = Image.fromarray(img_arr)
                img.save(result_img_path)

if __name__ == "__main__":
    main()



