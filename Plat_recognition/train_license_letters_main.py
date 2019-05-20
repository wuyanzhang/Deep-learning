from train_license_letters import *
from get_batch import *

batch_size=60

if __name__ == '__main__' and sys.argv[1] == 'train':
    # 获取图片总数
    input_count = 0
    for i in range(10,NUM_CLASSES+10):
        dir = 'E:/tensorflow_model/plat_recognition/dataset/train_images/training-set/letters/%s/' % i
        for files in os.listdir(dir):
            input_count += 1
    # 新建矩阵保存所有图片信息
    input_images = np.array([[0]*SIZE for i in range(input_count)])
    input_labels = np.array([[0]*NUM_CLASSES for i in range(input_count)])

    # 预处理数据,训练集
    index = 0
    for i in range(10,NUM_CLASSES+10):
        dir = 'E:/tensorflow_model/plat_recognition/dataset/train_images/training-set/letters/%s/' % i
        for files in os.listdir(dir):
            file = dir + files
            img = Image.open(file)
            width = img.size[0]
            height = img.size[1]
            for h in range(0,height):
                for w in range(0,width):
                    # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                    if img.getpixel((w, h)) > 230:
                        input_images[index][w + h * width] = 0
                    else:
                        input_images[index][w + h * width] = 1
            input_labels[index][i-10] = 1
            index += 1
    # 预处理数据,验证集
    val_count = 0
    for i in range(10, NUM_CLASSES+10):
        dir = 'E:/tensorflow_model/plat_recognition/dataset/train_images/validation-set/%s/' % i
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                val_count += 1

    # 定义对应维数和各维长度的数组
    val_images = np.array([[0] * SIZE for i in range(val_count)])
    val_labels = np.array([[0] * NUM_CLASSES for i in range(val_count)])

    # 第二次遍历图片目录是为了生成图片数据和标签
    index = 0
    for i in range(10, NUM_CLASSES+10):
        dir = 'E:/tensorflow_model/plat_recognition/dataset/train_images/validation-set/%s/' % i
        for files in os.listdir(dir):
            file = dir + files
            img = Image.open(file)
            width = img.size[0]
            height = img.size[1]
            for h in range(0, height):
                for w in range(0, width):
                    # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                    if img.getpixel((w, h)) > 230:
                        val_images[index][w + h * width] = 0
                    else:
                        val_images[index][w + h * width] = 1
            val_labels[index][i-10] = 1
            index += 1

    with tf.Session() as sess:
        # 第一层卷积
        W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev=0.1), name='W_conv1')
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]), name='b_conv1')
        conv_stride = [1, 1, 1, 1]
        kernel_size = [1, 2, 2, 1]
        pool_stride = [1, 2, 2, 1]
        L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_stride, kernel_size, pool_stride, padding='SAME')

        # 第二层卷积
        W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1), name='W_conv2')
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv2')
        conv_stride = [1, 1, 1, 1]
        kernel_size = [1, 1, 1, 1]
        pool_stride = [1, 1, 1, 1]
        L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_stride, kernel_size, pool_stride, padding='SAME')

        # 全连接层
        W_fc1 = tf.Variable(tf.truncated_normal([16 * 20 * 32, 512], stddev=0.1), name='W_fc1')
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]), name='b_fc1')
        h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20 * 32])
        h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)

        # dropout
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # 输出层(全连接层)
        W_fc2 = tf.Variable(tf.truncated_normal([512, NUM_CLASSES], stddev=0.1), name='W_fc2')
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name='b_fc2')
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # 定义优化器和训练op
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
        loss = tf.reduce_mean(cross_entropy, name="loss")
        train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess.run(tf.global_variables_initializer())

        time_elapsed = time.time() - time_begin
        print('Time spent reading image files: %f seconds.' % time_elapsed)
        time_begin = time.time()
        print('A total of %s training images were read, %s labels were read.' % (input_count,input_count))
        batch_count,remainder = get_batch(batch_size,input_count)

        # 执行训练迭代
        for it in range(iterations):
            for n in range(batch_count):
                sess.run(train_step,feed_dict={x:input_images[n*batch_size:(n+1)*batch_size],y:input_labels[n*batch_size:(n+1)*batch_size],keep_prob:0.5})
            if remainder > 0:
                start_index = batch_count*batch_size
                sess.run(train_step,feed_dict={x:input_images[start_index:input_count-1],y:input_labels[start_index:input_count-1],keep_prob:0.5})

            # 每五次迭代,输出一次准确率,准确率达到100%时退出循环
            if it % 5 == 0:
                val_accuracy = sess.run(accuracy,feed_dict={x:val_images,y:val_labels,keep_prob:1})
                print('Step %d: Accuracy: %0.5f ' % (it,val_accuracy))
                if val_accuracy > 0.95 and it >= 150:
                    break

        print('Finish training.')
        time_elapsed = time.time() - time_begin
        print("Time spent：%f seconds" % time_elapsed)
        time_begin = time.time()

        # 保存训练结果
        if not os.path.exists(SAVER_DIR):
            print('Create a new directory.')
            os.makedirs(SAVER_DIR)
        saver = tf.train.Saver()
        saver_path = saver.save(sess,'%smodel.ckpt' % (SAVER_DIR))

if __name__ == '__main__' and sys.argv[1] == 'predict':
    saver = tf.train.import_meta_graph('%smodel.ckpt.meta' % (SAVER_DIR))
    with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint(SAVER_DIR)
        saver.restore(sess,model_file)
        # 第一个卷积层
        W_conv1 = sess.graph.get_tensor_by_name("W_conv1:0")
        b_conv1 = sess.graph.get_tensor_by_name("b_conv1:0")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 2, 2, 1]
        pool_strides = [1, 2, 2, 1]
        L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')

        # 第二个卷积层
        W_conv2 = sess.graph.get_tensor_by_name("W_conv2:0")
        b_conv2 = sess.graph.get_tensor_by_name("b_conv2:0")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 1, 1, 1]
        pool_strides = [1, 1, 1, 1]
        L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')

        # 全连接层
        W_fc1 = sess.graph.get_tensor_by_name("W_fc1:0")
        b_fc1 = sess.graph.get_tensor_by_name("b_fc1:0")
        h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20 * 32])
        h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)

        # dropout
        keep_prob = tf.placeholder(tf.float32)

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # readout层
        W_fc2 = sess.graph.get_tensor_by_name("W_fc2:0")
        b_fc2 = sess.graph.get_tensor_by_name("b_fc2:0")

        # 定义优化器和训练op
        conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        for n in range(2, 3):
            path = "E:/tensorflow_model/plat_recognition/dataset/test_images/%s.bmp" % (n)
            img = Image.open(path)
            width = img.size[0]
            height = img.size[1]

            img_data = [[0] * SIZE for i in range(1)]
            for h in range(0, height):
                for w in range(0, width):
                    if img.getpixel((w, h)) < 190:
                        img_data[0][w + h * width] = 1
                    else:
                        img_data[0][w + h * width] = 0

            result = sess.run(conv, feed_dict={x: np.array(img_data), keep_prob: 1.0})
            max1 = 0
            max2 = 0
            max3 = 0
            max1_index = 0
            max2_index = 0
            max3_index = 0
            for j in range(NUM_CLASSES):
                if result[0][j] > max1:
                    max1 = result[0][j]
                    max1_index = j
                    continue
                if (result[0][j] > max2) and (result[0][j] <= max1):
                    max2 = result[0][j]
                    max2_index = j
                    continue
                if (result[0][j] > max3) and (result[0][j] <= max2):
                    max3 = result[0][j]
                    max3_index = j
                    continue

            if n == 3:
                license_num += "-"
            license_num = license_num + LETTERS_DIGITS[max1_index]
            print("概率：  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (
            LETTERS_DIGITS[max1_index], max1 * 100, LETTERS_DIGITS[max2_index], max2 * 100, LETTERS_DIGITS[max3_index],
            max3 * 100))

        print("城市代号是: 【%s】" % license_num)















