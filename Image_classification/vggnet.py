import tensorflow as tf

def inference(images,n_classes,batch_size):
    # 第一段卷积
    W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1), name='W_conv1')
    b_conv1 = tf.Variable(tf.constant(0.0, shape=[64]), name='b_conv1')
    conv_1 = tf.nn.conv2d(images, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
    pre_activation1 = tf.nn.bias_add(conv_1, b_conv1)
    conv1 = tf.nn.relu(pre_activation1, name='conv1')

    W_conv1_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1), name='W_conv1_1')
    b_conv1_2 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv1_2')
    conv_1_2 = tf.nn.conv2d(conv1, W_conv1_2, strides=[1, 1, 1, 1], padding='SAME')
    pre_activation1_2 = tf.nn.bias_add(conv_1_2, b_conv1_2)
    conv1_2 = tf.nn.relu(pre_activation1_2, name='conv1_2')

    # 第一段池化
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # 第二段卷积
    W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1), name='W_conv2')
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[128]), name='b_conv2')
    conv_2 = tf.nn.conv2d(pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    pre_activation2 = tf.nn.bias_add(conv_2, b_conv2)
    conv2 = tf.nn.relu(pre_activation2, name='conv2')

    W_conv2_2 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1), name='W_conv2_2')
    b_conv2_2 = tf.Variable(tf.constant(0.1, shape=[128]), name='b_conv2_2')
    conv_2_2 = tf.nn.conv2d(conv2, W_conv2_2, strides=[1, 1, 1, 1], padding='SAME')
    pre_activation2_2 = tf.nn.bias_add(conv_2_2, b_conv2_2)
    conv2_2 = tf.nn.relu(pre_activation2_2, name='conv2_2')

    # 第二段池化
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # 第三段卷积
    W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1), name='W_conv3')
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[256]), name='b_conv3')
    conv_3 = tf.nn.conv2d(pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
    pre_activation3 = tf.nn.bias_add(conv_3, b_conv3)
    conv3 = tf.nn.relu(pre_activation3, name='conv3')

    W_conv3_2 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1), name='W_conv3_2')
    b_conv3_2 = tf.Variable(tf.constant(0.1, shape=[256]), name='b_conv3_2')
    conv_3_2 = tf.nn.conv2d(conv3, W_conv3_2, strides=[1, 1, 1, 1], padding='SAME')
    pre_activation3_2 = tf.nn.bias_add(conv_3_2, b_conv3_2)
    conv3_2 = tf.nn.relu(pre_activation3_2, name='conv3_2')

    W_conv3_3 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1), name='W_conv3_3')
    b_conv3_3 = tf.Variable(tf.constant(0.1, shape=[256]), name='b_conv3_3')
    conv_3_3 = tf.nn.conv2d(conv3_2, W_conv3_3, strides=[1, 1, 1, 1], padding='SAME')
    pre_activation3_3 = tf.nn.bias_add(conv_3_3, b_conv3_3)
    conv3_3 = tf.nn.relu(pre_activation3_3, name='conv3_3')

    # 第三段池化
    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    # # 第四段卷积
    # W_conv4 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1), name='W_conv4')
    # b_conv4 = tf.Variable(tf.constant(0.1, shape=[512]), name='b_conv4')
    # conv_4 = tf.nn.conv2d(pool3, W_conv4, strides=[1, 1, 1, 1], padding='SAME')
    # pre_activation4 = tf.nn.bias_add(conv_4, b_conv4)
    # conv4 = tf.nn.relu(pre_activation4, name='conv4')
    #
    # W_conv4_2 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1), name='W_conv4_2')
    # b_conv4_2 = tf.Variable(tf.constant(0.1, shape=[512]), name='b_conv4_2')
    # conv_4_2 = tf.nn.conv2d(conv4, W_conv4_2, strides=[1, 1, 1, 1], padding='SAME')
    # pre_activation4_2 = tf.nn.bias_add(conv_4_2, b_conv4_2)
    # conv4_2 = tf.nn.relu(pre_activation4_2, name='conv4_2')
    #
    # W_conv4_3 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1), name='W_conv4_3')
    # b_conv4_3 = tf.Variable(tf.constant(0.1, shape=[512]), name='b_conv4_3')
    # conv_4_3 = tf.nn.conv2d(conv4_2, W_conv4_3, strides=[1, 1, 1, 1], padding='SAME')
    # pre_activation4_3 = tf.nn.bias_add(conv_4_3, b_conv4_3)
    # conv4_3 = tf.nn.relu(pre_activation4_3, name='conv4_3')
    #
    # # 第四段池化
    # pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    #
    # # 第五段卷积
    # W_conv5 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1), name='W_conv5')
    # b_conv5 = tf.Variable(tf.constant(0.1, shape=[512]), name='b_conv5')
    # conv_5 = tf.nn.conv2d(pool4, W_conv5, strides=[1, 1, 1, 1], padding='SAME')
    # pre_activation5 = tf.nn.bias_add(conv_5, b_conv5)
    # conv5 = tf.nn.relu(pre_activation5, name='conv5')
    #
    # W_conv5_2 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1), name='W_conv5_2')
    # b_conv5_2 = tf.Variable(tf.constant(0.1, shape=[512]), name='b_conv5_2')
    # conv_5_2 = tf.nn.conv2d(conv5, W_conv5_2, strides=[1, 1, 1, 1], padding='SAME')
    # pre_activation5_2 = tf.nn.bias_add(conv_5_2, b_conv5_2)
    # conv5_2 = tf.nn.relu(pre_activation5_2, name='conv5_2')
    #
    # W_conv5_3 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1), name='W_conv5_3')
    # b_conv5_3 = tf.Variable(tf.constant(0.1, shape=[512]), name='b_conv5_3')
    # conv_5_3 = tf.nn.conv2d(conv5_2, W_conv5_3, strides=[1, 1, 1, 1], padding='SAME')
    # pre_activation5_3 = tf.nn.bias_add(conv_5_3, b_conv5_3)
    # conv5_3 = tf.nn.relu(pre_activation5_3, name='conv5_3')
    #
    # # 第五段池化
    # pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

    # 第一层全连接
    reshape = tf.reshape(pool3, shape=[batch_size, -1])
    dim = reshape.get_shape()[1].value
    W_fc1 = tf.Variable(tf.truncated_normal([dim, 128], stddev=0.00005), name='W_fc1')
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[128]), name='b_fc1')
    fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1, name='fc1')

    # 第二层全连接
    W_fc2 = tf.Variable(tf.truncated_normal([128, 128], stddev=0.00005), name='W_fc2')
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[128]), name='b_fc2')
    fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2) + b_fc2, name='fc2')

    # 最后一个softmax层
    W_output = tf.Variable(tf.truncated_normal([128, n_classes], stddev=0.00005), name='W_output')
    b_output = tf.Variable(tf.constant(0.1, shape=[n_classes]), name='b_output')
    output = tf.nn.softmax(tf.matmul(fc2, W_output) + b_output)
    return output

def losses(logits,labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy')
    loss = tf.reduce_mean(cross_entropy,name='loss')
    return loss

def training(loss,learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_step = optimizer.minimize(loss,global_step=global_step)
    return train_step

def evaluation(logits,labels):
    # correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    correct = tf.nn.in_top_k(logits, labels, 1)
    correct = tf.cast(correct, tf.float16)
    accuracy = tf.reduce_mean(correct)
    return accuracy