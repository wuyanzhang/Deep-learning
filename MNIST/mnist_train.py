import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data",one_hot=True)

batch_size = 100
learning_rate = 0.01
learning_rate_decay = 0.999
max_steps = 30000

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

regularizer = tf.contrib.layers.l2_regularizer(0.0001)

y = hidden_layer(x,regularizer)

training_step = tf.Variable(0,trainable=False)
average_class = tf.train.ExponentialMovingAverage(0.99,training_step)
average_op = average_class.apply(tf.trainable_variables())

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))

loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection("losses"))

learning_rate = tf.train.exponential_decay(learning_rate,training_step,mnist.train.num_examples/batch_size,learning_rate_decay)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=training_step)

with tf.control_dependencies([train_step,average_op]):
	# tf.no_op()表示不执行任何操作
	train_op = tf.no_op(name="train")

saver = tf.train.Saver()
with tf.Session() as sess:
	tf.global_variables_initializer().run()
	for i in range(max_steps):
		x_train,y_train = mnist.train.next_batch(batch_size)
		_,loss_value,step = sess.run([train_op,loss,training_step],feed_dict={x:x_train,y_:y_train})

		if i % 1000 == 0:
			print("After %d training steps,loss on training batch is %g." % (step,loss_value))
			saver.save(sess,"./checkpoint/mnist_model.ckpt",global_step=training_step)
