import tensorflow as tf

def inference(images,n_classes,batch_size):
    # 第一层卷积
    W_conv1 = tf.Variable(tf.truncated_normal([3,3,3,16],stddev=0.1),name='W_conv1')
    b_conv1 = tf.Variable(tf.constant(0.1,shape=[16]),name='b_conv1')
    conv_1 = tf.nn.conv2d(images,W_conv1,strides=[1,1,1,1],padding='SAME')
    pre_activation1 = tf.nn.bias_add(conv_1,b_conv1)
    conv1 = tf.nn.relu(pre_activation1,name='conv1')

    # 第一层池化
    pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1')
    norm1 = tf.nn.lrn(pool1,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm1')

    # 第二层卷积
    W_conv2 = tf.Variable(tf.truncated_normal([3,3,16,16],stddev=0.1),name='W_conv2')
    b_conv2 = tf.Variable(tf.constant(0.1,shape=[16]),name='b_conv2')
    conv_2 = tf.nn.conv2d(norm1,W_conv2,strides=[1,1,1,1],padding='SAME')
    pre_activation2 = tf.nn.bias_add(conv_2,b_conv2)
    conv2 = tf.nn.relu(pre_activation2,name='conv2')

    # 第二层池化
    pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool2')
    norm2 = tf.nn.lrn(pool2,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm2')

    # 第一层全连接
    reshape = tf.reshape(norm2, shape=[batch_size, -1])
    dim = reshape.get_shape()[1].value
    W_fc1 = tf.Variable(tf.truncated_normal([dim,128],stddev=0.005),name='W_fc1')
    b_fc1 = tf.Variable(tf.constant(0.1,shape=[128]),name='b_fc1')
    fc1 = tf.nn.relu(tf.matmul(reshape,W_fc1) + b_fc1,name='fc1')

    # 第二层全连接
    W_fc2 = tf.Variable(tf.truncated_normal([128,128],stddev=0.005),name='W_fc2')
    b_fc2 = tf.Variable(tf.constant(0.1,shape=[128]),name='b_fc2')
    fc2 = tf.nn.relu(tf.matmul(fc1,W_fc2) + b_fc2,name='fc2')

    # 最后一个softmax层
    W_output = tf.Variable(tf.truncated_normal([128,n_classes],stddev=0.005),name='W_output')
    b_output = tf.Variable(tf.constant(0.1,shape=[n_classes]),name='b_output')
    output = tf.nn.softmax(tf.matmul(fc2,W_output) + b_output)
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



















