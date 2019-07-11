import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data",one_hot=True)

def hidden_layer(input,regularizer):
	with tf.variable_scope("hidden_layer"):
		weights = tf.get_variable("weights",[784,500],initializer=tf.truncated_normal_initializer(stddev=0.1))

		if regularizer != None:
			tf.add_to_collection("losses",regularizer(weights))
		bias = tf.get_variable("bias",[500],initializer=tf.constant_initializer(0.0))
		hidden_layer = tf.nn.relu(tf.matmul(input,weights) + bias)

	with tf.variable_scope("hidden_layer_output"):
		weights = tf.get_variable("weights",[500,10],initializer=tf.truncated_normal_initializer(stddev=0.1))

		if regularizer != None:
			tf.add_to_collection("losses",regularizer(weights))
		bias = tf.get_variable("bias",[10],initializer=tf.constant_initializer(0.0))
		hidden_layer_output = tf.matmul(hidden_layer,weights) + bias
	return hidden_layer_output

x = tf.placeholder(tf.float32,[None,784],name="input_x")
y_ = tf.placeholder(tf.float32,[None,10],name="input_y")

y = hidden_layer(x,None)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 通过使用variables_to_restore函数，可以使在加载模型的时候将影子变量直接映射到变量的本身，所以我们在获取变量的滑动平均值的时候只需要获取到变量的本身值而不需要去获取影子变量
variable_averages = tf.train.ExponentialMovingAverage(0.99)
saver = tf.train.Saver(variable_averages.variables_to_restore())

with tf.Session() as sess:
	# 准备验证集数据
	validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
	# 准备测试集数据
	test_feed = {x:mnist.test.images,y_:mnist.test.labels}

	# get_checkpoint_state()函数会自动找到目录中最新模型的文件名
	ckpt = tf.train.get_checkpoint_state("./checkpoint/")

	# 加载模型
	saver.restore(sess,ckpt.model_checkpoint_path)

	# 通过文件名得到模型保存时迭代的轮数
	global_step = ckpt.model_checkpoint_path.split('-')[-1]
	print("The latest ckpt is mnist_model.ckpt-%s" % (global_step))

	# 计算在验证集上的准确率
	accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
	print("After %s training steps,validation accuracy = %g%%" % (global_step,accuracy_score * 100))

	# 计算在测试集上的准确率
	test_accuracy = sess.run(accuracy,feed_dict=test_feed)
	print("After %s training steps,test accuracy = %g%%" % (global_step,test_accuracy * 100))