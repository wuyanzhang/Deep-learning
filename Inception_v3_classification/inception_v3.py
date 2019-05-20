import tensorflow as tf
import os
from input_data_process import *
from tensorflow.python.platform import gfile

# 定义保存模型的路径以及模型文件的名字
model_path = "./inception_model"
model_name = "classify_image_graph_def.pb"

num_steps = 4000
BATCH_SIZE = 100

# bottleneck_size是inception_v3模型瓶颈层结点的数量
bottleneck_size = 2048

# 字典{类名:信息}
image_lists = create_image_dict()

# 分类数
num_classes = len(image_lists.keys())

# 读取已经训练好的Inception_v3模型
# with gfile.FastGFile(os.path.join(model_path,model_name)) as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())

with tf.gfile.FastGFile(os.path.join(model_path,model_name), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

bottleneck_tensor,jepg_data_tensor = tf.import_graph_def(graph_def,return_elements=["pool_3/_reshape:0","DecodeJpeg/contents:0"])

x = tf.placeholder(tf.float32,[None,bottleneck_size])
y = tf.placeholder(tf.float32,[None,num_classes])

# 定义一层全连接层
with tf.name_scope("Final_training_op"):
    weights = tf.Variable(tf.truncated_normal([bottleneck_size,num_classes],stddev=0.001))
    biases = tf.Variable(tf.zeros([num_classes]))
    logits = tf.matmul(x,weights) + biases
    final_tensor = tf.nn.softmax(logits)

# 定义交叉熵损失函数以及train_step使用的随机梯度下降优化器
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_mean)

# 定义计算正确率
correct_prediction = tf.equal(tf.arg_max(final_tensor,1),tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(num_steps):
        # 使用get_batch()函数产生一个批次训练用的随机特征向量数据及其对应的label
        train_bottlenecks,train_labels = get_batch(sess,num_classes,image_lists,BATCH_SIZE,
                                                   "training",jepg_data_tensor,bottleneck_tensor)
        sess.run(train_step,feed_dict={x:train_bottlenecks,y:train_labels})

        if i % 100 == 0:
            validation_bottlenecks,validation_labels = get_batch(sess,num_classes,image_lists,BATCH_SIZE,
                                                                "validation",jepg_data_tensor,bottleneck_tensor)
            validation_accuracy = sess.run(accuracy,feed_dict={x:validation_bottlenecks,y:validation_labels})

            print("Step %d: Validation accuracy = %.4f%%" % (i,validation_accuracy * 100))

    # 在最后的测试数据上测试正确率，这里调用get_test_bottlenecks(),返回所有图片的特征向量作为特征数据
    test_bottlenecks,test_labels = get_test_bottlenecks(sess,image_lists,num_classes,jepg_data_tensor,bottleneck_tensor)
    test_accuracy = sess.run(accuracy,feed_dict={x:test_bottlenecks,y:test_labels})
    print("Finally test accuracy = %.4f%%" % (test_accuracy * 100))

















