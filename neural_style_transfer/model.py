import tensorflow as tf
import time

# VGG 自带的一个常量，之前VGG训练通过归一化，所以现在同样需要作此操作
VGG_MEAN = [103.939, 116.779, 123.68] # rgb 三通道的均值

class VGGNet():
    '''
    创建 vgg16 网络 结构
    从模型中载入参数
    '''
    def __init__(self, data_dict):
        '''
        传入vgg16模型
        :param data_dict: vgg16.npy (字典类型)
        '''
        self.data_dict = data_dict
    
    def get_conv_filter(self, name):
        '''
        得到对应名称的卷积层
        :param name: 卷积层名称
        :return: 该卷积层输出
        '''
        return tf.constant(self.data_dict[name][0],name = "conv")
    
    def get_fc_weight(self, name):
        '''
        获得名字为name的全连接层权重
        :param name: 连接层名称
        :return: 该层权重
        '''
        return tf.constant(self.data_dict[name][0], name = 'fc')

    def get_bias(self, name):
        '''
        获得名字为name的全连接层偏置
        :param name: 连接层名称
        :return: 该层偏置
        '''
        return tf.constant(self.data_dict[name][1], name = 'bias')

    def conv_layer(self, x, name):
        '''
        创建一个卷积层
        :param x:
        :param name:
        :return:
        '''
        with tf.name_scope(name):
            conv_w = self.get_conv_filter(name)
            conv_b = self.get_bias(name)
            h = tf.nn.conv2d(x, conv_w, strides=[1, 1, 1, 1], padding = 'SAME')
            h = tf.nn.bias_add(h, conv_b)
            h = tf.nn.relu(h)
            return h
    
    def pooling_layer(self, x, name):
        '''
        创建池化层
        :param x: 输入的tensor
        :param name: 池化层名称
        :return: tensor
        '''
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME', name = name)    

    def flatten_layer(self, x, name):
        '''
        展平
        :param x: input_tensor
        :param name:
        :return: 二维矩阵
        '''
        with tf.name_scope(name):
            x_shape = x.get_shape().as_list()
            dim = 1
            for d in x_shape[1:]:
                dim *= d
            x = tf.reshape(x,[-1, dim])
            return x

    def build(self, x_rgb):
        '''
        创建vgg16 网络
        :param x_rgb: [1, 224, 224, 3]
        :return:
        '''
        start_time = time.time()
        print("开始加载模型")
        # 将输入图像进行处理，将每个通道减去均值
        r, g, b = tf.split(x_rgb, [1, 1, 1], axis = 3)
        '''
        tf.split(value, num_or_size_split, axis=0)用法：
        value:输入的Tensor
        num_or_size_split:有两种用法：
            1.直接传入一个整数，代表会被切成几个张量，切割的维度有axis指定
            2.传入一个向量，向量长度就是被切的份数。传入向量的好处在于，可以指定每一份有多少元素
        axis, 指定从哪一个维度切割
        因此，上一句的意思就是从第4维切分，分为3份，每一份只有1个元素
        '''

        x_bgr = tf.concat([b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=3)

        # 开始构建卷积层
        # vgg16 的网络结构
        # 第一层：2个卷积层 1个pooling层
        # 第二层：2个卷积层 1个pooling层
        # 第三层：3个卷积层 1个pooling层
        # 第四层：3个卷积层 1个pooling层
        # 第五层：3个卷积层 1个pooling层
        # 这些变量名称不能乱取，必须要和vgg16模型保持一致
        # 另外，将这些卷积层用self.的形式，方便以后取用方便
        self.conv1_1 = self.conv_layer(x_bgr, 'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2')
        self.pool1 = self.pooling_layer(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
        self.pool2 = self.pooling_layer(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        self.pool3 = self.pooling_layer(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        self.pool4 = self.pooling_layer(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')
        self.pool5 = self.pooling_layer(self.conv5_3, 'pool5')

        print("模型加载结束：%4f" % (time.time() - start_time))