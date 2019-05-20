import tensorflow as tf
import tensorflow.contrib.slim as slim

# 定义简单的函数产生截断的正态分布
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

# 定义函数 inception_v3_arg_scope 用来生成网络中经常用到的函数的默认参数
def inception_v3_arg_scope(weight_decay=0.00004, stddev=0.1,batch_norm_var_collection="moving_vars"):
    batch_norm_params = {
        "decay": 0.9997, "epsilon": 0.001, "updates_collections": tf.GraphKeys.UPDATE_OPS,
        "variables_collections": {
            "beta": None, "gamma": None, "moving_mean": [batch_norm_var_collection],
            "moving_variance": [batch_norm_var_collection]
        }
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],weights_regularizer=slim.l2_regularizer(weight_decay)):
        # 对卷积层生成函数的几个参数赋予默认值
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=tf.truncated_normal_initializer(stddev=stddev),
                            activation_fc=tf.nn.relu,
                            normalizer_fc=slim.batch_norm,
                            normalizer_params=batch_norm_params) as scope:
    return scope

# 定义inception_v3的卷积部分
def inception_v3_base(inputs,scope=None):
    end_points = {}
    with tf.variable_scope(scope,"Inception_v3",[inputs]):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding="VALID"):
            net = slim.conv2d(inputs,num_outputs=32,kernel_size=[3,3],stride=2,scope="Conv2d_1a_3x3")
            net = slim.conv2d(net,num_outputs=32,kernel_size=[3,3],scope="Conv2d_2a_3x3")
            net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],padding="SAME",scope="Conv2d_2b_3x3")
            net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope="MaxPool_3a_3x3")
            net = slim.conv2d(net,num_outputs=80,kernel_size=[1,1],scope="Conv2d_3b_1x1")
            net = slim.conv2d(net,num_outputs=192,kernel_size=[3,3],scope="Conv2d_4a_3x3")
            net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope="MaxPool_5a_3x3")

        # 定义第一个inception模块组
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],sstride=1,padding="SAME"):
            # 定义第一个inception模块组中的第一个inception module
            with tf.variable_scope("Mixed_5b"):
                with tf.variable_scope("Branch_0"):
                    batch_0 = slim.conv2d(net,num_outputs=64,kernel_size=[1,1],scope="Conv2d_0a_1x1")
                with tf.variable_scope("Branch_1"):
                    batch_1 = slim.conv2d(net,num_outputs=48,kernel_size=[1,1],scope="Conv2d_0a_1x1")
                    batch_1 = slim.conv2d(batch_1,num_outputs=64,kernel_size=[5,5],scope="Conv2d_0b_5x5")
                with tf.variable_scope("Branch_2"):
                    batch_2 = slim.conv2d(net,num_outputs=64,kernel_size=[1,1],scope="Conv2d_0a_1x1")
                    batch_2 = slim.conv2d(batch_2,num_outputs=96,kernel_size=[3,3],scope="Conv2d_0b_3x3")
                    batch_2 = slim.conv2d(batch_2,num_outputs=96,kernel_size=[3,3],scope="Conv2d_0c_3x3")
                with tf.variable_scope("Branch_3"):
                    batch_3 = slim.avg_pool2d(net,kernel_size=[3,3],scope="AvgPool_0a_3x3")
                    batch_3 = slim.conv2d(batch_3,num_outputs=32,kernel_size=[1,1],scope="Conv2d_0b_1x1")
                net = tf.concat([batch_0,batch_1,batch_2,batch_3],3)

            # 定义第一个inception模块组中的第二个inception module
            with tf.variable_scope("Mixed_5c"):
                with tf.variable_scope("Branch_0"):
                    batch_0 = slim.conv2d(net, num_outputs=64, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                with tf.variable_scope("Branch_1"):
                    batch_1 = slim.conv2d(net, num_outputs=48, kernel_size=[1, 1], scope="Conv2d_0b_1x1")
                    batch_1 = slim.conv2d(batch_1, num_outputs=64, kernel_size=[5, 5], scope="Conv2d_0c_5x5")
                with tf.variable_scope("Branch_2"):
                    batch_2 = slim.conv2d(net, num_outputs=64, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=96, kernel_size=[3, 3], scope="Conv2d_0b_3x3")
                    batch_2 = slim.conv2d(batch_2, num_outputs=96, kernel_size=[3, 3], scope="Conv2d_0c_3x3")
                with tf.variable_scope("Branch_3"):
                    batch_3 = slim.avg_pool2d(net, kernel_size=[3, 3], scope="AvgPool_0a_3x3")
                    batch_3 = slim.conv2d(batch_3, num_outputs=64, kernel_size=[1, 1], scope="Conv2d_0b_1x1")
                net = tf.concat([batch_0, batch_1, batch_2, batch_3], 3)

            # 定义第一个inception模块组中的第三个inception module
            with tf.variable_scope("Mixed_5c"):
                with tf.variable_scope("Branch_0"):
                    batch_0 = slim.conv2d(net, num_outputs=64, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                with tf.variable_scope("Branch_1"):
                    batch_1 = slim.conv2d(net, num_outputs=48, kernel_size=[1, 1], scope="Conv2d_0b_1x1")
                    batch_1 = slim.conv2d(batch_1, num_outputs=64, kernel_size=[5, 5], scope="Conv2d_0c_5x5")
                with tf.variable_scope("Branch_2"):
                    batch_2 = slim.conv2d(net, num_outputs=64, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=96, kernel_size=[3, 3], scope="Conv2d_0b_3x3")
                    batch_2 = slim.conv2d(batch_2, num_outputs=96, kernel_size=[3, 3], scope="Conv2d_0c_3x3")
                with tf.variable_scope("Branch_3"):
                    batch_3 = slim.avg_pool2d(net, kernel_size=[3, 3], scope="AvgPool_0a_3x3")
                    batch_3 = slim.conv2d(batch_3, num_outputs=64, kernel_size=[1, 1], scope="Conv2d_0b_1x1")
                net = tf.concat([batch_0, batch_1, batch_2, batch_3], 3)

            # 定义第二个inception模块组中的第一个inception module
            with tf.variable_scope("Mixed_6a"):
                with tf.variable_scope("Branch_0"):
                    batch_0 = slim.conv2d(net, num_outputs=384, kernel_size=[3, 3],
                                          stride=2, padding="VALID", scope="Conv2d_1a_1x1")
                with tf.variable_scope("Branch_1"):
                    batch_1 = slim.conv2d(net, num_outputs=64, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_1 = slim.conv2d(batch_1, num_outputs=96, kernel_size=[3, 3], scope="Conv2d_0b_3x3")
                    batch_1 = slim.conv2d(batch_1, num_outputs=96, kernel_size=[3, 3],
                                          stride=2, padding="VALID", scope="Conv2d_1a_1x1")
                with tf.variable_scope("Branch_2"):
                    batch_2 = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding="VALID",
                                              scope="MaxPool_1a_3x3")

                net = tf.concat([batch_0, batch_1, batch_2], 3)

            # 定义第二个inception模块组中的第二个inception module
            with tf.variable_scope("Mixed_6b"):
                with tf.variable_scope("Branch_0"):
                    batch_0 = slim.conv2d(net, num_outputs=192, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                with tf.variable_scope("Branch_1"):
                    batch_1 = slim.conv2d(net, num_outputs=128, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_1 = slim.conv2d(batch_1, num_outputs=128, kernel_size=[1, 7], scope="Conv2d_0b_1x7")
                    batch_1 = slim.conv2d(batch_1, num_outputs=192, kernel_size=[7, 1], scope="Conv2d_0c_7x1")
                with tf.variable_scope("Branch_2"):
                    batch_2 = slim.conv2d(net, num_outputs=128, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=128, kernel_size=[7, 1], scope="Conv2d_0b_7x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=128, kernel_size=[1, 7], scope="Conv2d_0c_1x7")
                    batch_2 = slim.conv2d(batch_2, num_outputs=128, kernel_size=[7, 1], scope="Conv2d_0d_7x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=192, kernel_size=[1, 7], scope="Conv2d_0e_1x7")
                with tf.variable_scope("Branch_3"):
                    batch_3 = slim.avg_pool2d(net, kernel_size=[3, 3], scope="AvgPool_0a_3x3")
                    batch_3 = slim.conv2d(batch_3, num_outputs=192, kernel_size=[1, 1], scope="Conv2d_0b_1x1")

                net = tf.concat([batch_0, batch_1, batch_2, batch_3], 3)

            # 定义第二个inception模块组中的第三个inception module
            with tf.variable_scope("Mixed_6c"):
                with tf.variable_scope("Branch_0"):
                    batch_0 = slim.conv2d(net, num_outputs=192, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                with tf.variable_scope("Branch_1"):
                    batch_1 = slim.conv2d(net, num_outputs=160, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_1 = slim.conv2d(batch_1, num_outputs=160, kernel_size=[1, 7], scope="Conv2d_0b_1x7")
                    batch_1 = slim.conv2d(batch_1, num_outputs=160, kernel_size=[7, 1], scope="Conv2d_0c_7x1")
                with tf.variable_scope("Branch_2"):
                    batch_2 = slim.conv2d(net, num_outputs=160, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=160, kernel_size=[7, 1], scope="Conv2d_0b_7x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=160, kernel_size=[1, 7], scope="Conv2d_0c_1x7")
                    batch_2 = slim.conv2d(batch_2, num_outputs=160, kernel_size=[7, 1], scope="Conv2d_0d_7x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=192, kernel_size=[1, 7], scope="Conv2d_0e_1x7")
                with tf.variable_scope("Branch_3"):
                    batch_3 = slim.avg_pool2d(net, kernel_size=[3, 3], scope="AvgPool_0a_3x3")
                    batch_3 = slim.conv2d(batch_3, num_outputs=192, kernel_size=[1, 1], scope="Conv2d_0b_1x1")

                net = tf.concat([batch_0, batch_1, batch_2, batch_3], 3)

            # 定义第二个inception模块组中的第四个inception module
            with tf.variable_scope("Mixed_6d"):
                with tf.variable_scope("Branch_0"):
                    batch_0 = slim.conv2d(net, num_outputs=192, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                with tf.variable_scope("Branch_1"):
                    batch_1 = slim.conv2d(net, num_outputs=160, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_1 = slim.conv2d(batch_1, num_outputs=160, kernel_size=[1, 7], scope="Conv2d_0b_1x7")
                    batch_1 = slim.conv2d(batch_1, num_outputs=160, kernel_size=[7, 1], scope="Conv2d_0c_7x1")
                with tf.variable_scope("Branch_2"):
                    batch_2 = slim.conv2d(net, num_outputs=160, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=160, kernel_size=[7, 1], scope="Conv2d_0b_7x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=160, kernel_size=[1, 7], scope="Conv2d_0c_1x7")
                    batch_2 = slim.conv2d(batch_2, num_outputs=160, kernel_size=[7, 1], scope="Conv2d_0d_7x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=192, kernel_size=[1, 7], scope="Conv2d_0e_1x7")
                with tf.variable_scope("Branch_3"):
                    batch_3 = slim.avg_pool2d(net, kernel_size=[3, 3], scope="AvgPool_0a_3x3")
                    batch_3 = slim.conv2d(batch_3, num_outputs=192, kernel_size=[1, 1], scope="Conv2d_0b_1x1")

                net = tf.concat([batch_0, batch_1, batch_2, batch_3], 3)

            # 定义第二个inception模块组中的第五个inception module
            with tf.variable_scope("Mixed_6e"):
                with tf.variable_scope("Branch_0"):
                    batch_0 = slim.conv2d(net, num_outputs=192, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                with tf.variable_scope("Branch_1"):
                    batch_1 = slim.conv2d(net, num_outputs=160, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_1 = slim.conv2d(batch_1, num_outputs=160, kernel_size=[1, 7], scope="Conv2d_0b_1x7")
                    batch_1 = slim.conv2d(batch_1, num_outputs=160, kernel_size=[7, 1], scope="Conv2d_0c_7x1")
                with tf.variable_scope("Branch_2"):
                    batch_2 = slim.conv2d(net, num_outputs=160, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=160, kernel_size=[7, 1], scope="Conv2d_0b_7x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=160, kernel_size=[1, 7], scope="Conv2d_0c_1x7")
                    batch_2 = slim.conv2d(batch_2, num_outputs=160, kernel_size=[7, 1], scope="Conv2d_0d_7x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=192, kernel_size=[1, 7], scope="Conv2d_0e_1x7")
                with tf.variable_scope("Branch_3"):
                    batch_3 = slim.avg_pool2d(net, kernel_size=[3, 3], scope="AvgPool_0a_3x3")
                    batch_3 = slim.conv2d(batch_3, num_outputs=192, kernel_size=[1, 1], scope="Conv2d_0b_1x1")

                net = tf.concat([batch_0, batch_1, batch_2, batch_3], 3)

            end_points["Mixed_6e"] = net

            # 定义第三个inception模块组中的第一个inception module
            with tf.variable_scope("Mixed_7a"):
                with tf.variable_scope("Branch_0"):
                    batch_0 = slim.conv2d(net, num_outputs=192, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_0 = slim.conv2d(net, num_outputs=320, kernel_size=[3, 3], stride=2,
                                          padding="VALID", scope="Conv2d_1a_3x3")
                with tf.variable_scope("Branch_1"):
                    batch_1 = slim.conv2d(net, num_outputs=192, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_1 = slim.conv2d(batch_1, num_outputs=192, kernel_size=[1, 7], scope="Conv2d_0b_1x7")
                    batch_1 = slim.conv2d(batch_1, num_outputs=192, kernel_size=[7, 1], scope="Conv2d_0c_7x1")
                    batch_1 = slim.conv2d(batch_1, num_outputs=192, kernel_size=[3, 3], stride=2,
                                          padding="VALID", scope="Conv2d_1a_3x3")
                with tf.variable_scope("Branch_2"):
                    batch_2 = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding="VALID",
                                              scope="MaxPool_1a_3x3")

                net = tf.concat([batch_0, batch_1, batch_2], 3)

            # 定义第三个inception模块组中的第二个inception module
            with tf.variable_scope("Mixed_7b"):
                with tf.variable_scope("Branch_0"):
                    batch_0 = slim.conv2d(net, num_outputs=320, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                with tf.variable_scope("Branch_1"):
                    batch_1 = slim.conv2d(net, num_outputs=384, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_1 = tf.concat([
                        slim.conv2d(batch_1, num_outputs=384, kernel_size=[1, 3], scope="Conv2d_0b_1x3"),
                        slim.conv2d(batch_1, num_outputs=384, kernel_size=[3, 1], scope="Conv2d_0b_3x1")], axis=3)
                with tf.variable_scope("Branch_2"):
                    batch_2 = slim.conv2d(net, num_outputs=448, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=384, kernel_size=[3, 3], scope="Conv2d_0b_3x3")
                    batch_2 = tf.concat([
                        slim.conv2d(batch_2, num_outputs=384, kernel_size=[1, 3], scope="Conv2d_0c_1x3"),
                        slim.conv2d(batch_2, num_outputs=384, kernel_size=[3, 1], scope="Conv2d_0d_3x1")], axis=3)
                with tf.variable_scope("Branch_3"):
                    batch_3 = slim.avg_pool2d(net, kernel_size=[3, 3], scope="AvgPool_0a_3x3")
                    batch_3 = slim.conv2d(batch_3, num_outputs=192, kernel_size=[1, 1], scope="Conv2d_0b_1x1")

            net = tf.concat([batch_0, batch_1, batch_2, batch_3], 3)

            # 定义第三个inception模块组中的第三个inception module
            with tf.variable_scope("Mixed_7c"):
                with tf.variable_scope("Branch_0"):
                    batch_0 = slim.conv2d(net, num_outputs=320, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                with tf.variable_scope("Branch_1"):
                    batch_1 = slim.conv2d(net, num_outputs=384, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_1 = tf.concat([
                        slim.conv2d(batch_1, num_outputs=384, kernel_size=[1, 3], scope="Conv2d_0b_1x3"),
                        slim.conv2d(batch_1, num_outputs=384, kernel_size=[3, 1], scope="Conv2d_0b_3x1")], axis=3)
                with tf.variable_scope("Branch_2"):
                    batch_2 = slim.conv2d(net, num_outputs=448, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=384, kernel_size=[3, 3], scope="Conv2d_0b_3x3")
                    batch_2 = tf.concat([
                        slim.conv2d(batch_2, num_outputs=384, kernel_size=[1, 3], scope="Conv2d_0c_1x3"),
                        slim.conv2d(batch_2, num_outputs=384, kernel_size=[3, 1], scope="Conv2d_0d_3x1")], axis=3)
                with tf.variable_scope("Branch_3"):
                    batch_3 = slim.avg_pool2d(net, kernel_size=[3, 3], scope="AvgPool_0a_3x3")
                    batch_3 = slim.conv2d(batch_3, num_outputs=192, kernel_size=[1, 1], scope="Conv2d_0b_1x1")

            net = tf.concat([batch_0, batch_1, batch_2, batch_3], 3)

        return net,end_points

def inception_v3(inputs,num_classes=1000,is_training=True,droupot_keep_prob = 0.8,prediction_fn = slim.softmax,spatial_squeeze = True,reuse = None,scope="InceptionV3"):
    """
    InceptionV3整个网络的构建
    param :
    inputs -- 输入tensor
    num_classes -- 最后分类数目
    is_training -- 是否是训练过程
    droupot_keep_prob -- dropout保留节点比例
    prediction_fn -- 最后分类函数，默认为softmax
    patial_squeeze -- 是否对输出去除维度为1的维度
    reuse -- 是否对网络和Variable重复使用
    scope -- 函数默认参数环境
    return:
    logits -- 最后输出结果
    end_points -- 包含辅助节点的重要节点字典表
    """
    with tf.variable_scope(scope,"Inception_v3",[inputs,num_classes],reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm,slim.dropout],is_training=is_training):
            net,end_points = inception_v3_base(inputs,scope=scope)

            # 辅助分类结点部分
            with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding="SAME"):
                # 通过end_points取到Mixed_6e
                aux_logits = end_points["Mixed_6e"]
                with tf.variable_scope("AuxLogits"):
                    aux_logits = slim.avg_pool2d(aux_logits, kernel_size=[5, 5], stride=3,
                                                 padding="VALID", scope="Avgpool_1a_5x5")
                    aux_logits = slim.conv2d(aux_logits, num_outputs=128, kernel_size=[1, 1], scope="Conv2d_1b_1x1")
                    aux_logits = slim.conv2d(aux_logits, num_outputs=768, kernel_size=[5, 5],
                                             weights_initializer=trunc_normal(0.01), padding="VALID",
                                             scope="Conv2d_2a_5x5")
                    aux_logits = slim.conv2d(aux_logits, num_outputs=num_classes, kernel_size=[1, 1],
                                             activation_fn=None, normalizer_fn=None,
                                             weights_initializer=trunc_normal(0.001), scope="Conv2d_1b_1x1")

                    # 消除tensor中前两个维度为1的维度
                    if spatial_squeeze:
                        aux_logits = tf.squeeze(aux_logits,axis=[1,2],name="SpatialSqueeze")

                    end_points["AuxLogits"] = aux_logits  # 将辅助节点分类的输出aux_logits存到end_points中

                # 正常的分类预测
                with tf.variable_scope("Logits"):
                    net = slim.avg_pool2d(net,kernel_size=[8,8],padding="VALID",scope="Avgpool_1a_8x8")
                    net = slim.dropout(net, keep_prob=droupot_keep_prob, scope="Dropout_1b")
                    end_points["Logits"] = net

                    logits = slim.conv2d(net, num_outputs=num_classes, kernel_size=[1, 1], activation_fn=None,
                                         normalizer_fn=None, scope="Conv2d_1c_1x1")
                    if spatial_squeeze:
                        logits = tf.squeeze(logits, axis=[1, 2], name="SpatialSqueeze")

                end_points["Logits"] = logits
                end_points["Predictions"] = prediction_fn(logits, scope="Predictions")

        return logits,end_points















