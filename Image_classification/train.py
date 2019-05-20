import tensorflow as tf
from vggnet import *
from input_data_process import *

N_CLASSES = 5 #分类的类数
IMG_H = 224
IMG_W = 224
BATCH_SIZE = 10
CAPACITY = 2000
MAX_STEP = 5000 #最大迭代次数
learning_rate = 0.000001 #学习率
train_dir = 'E:/Project/tensorflow_model/AlexNet_classification/dataset/'
logs_dir = 'E:/Project/tensorflow_model/AlexNet_classification/logs_vggnet/'

def run_training():
    train,train_label,input_count = get_files(train_dir)
    train_image_batch,train_label_batch = get_batch(train,train_label,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)
    train_logits = inference(train_image_batch,N_CLASSES,BATCH_SIZE)
    train_loss = losses(train_logits,train_label_batch)
    train_step = training(train_loss,learning_rate)
    train_acc = evaluation(train_logits,train_label_batch)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                _, tra_loss, tra_acc = sess.run([train_step, train_loss, train_acc])

                if step % 100 == 0:
                    print("Step %d, train loss = %.2f, train accuracy = %.2f" % (step, tra_loss, tra_acc))
                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_dir, "model.ckpt")
                    saver.save(sess, checkpoint_path, global_step=step)
            # if not os.path.exists(logs_dir):
            #     print('Create a new directory.')
            #     os.makedirs(logs_dir)
            # saver = tf.train.Saver()
            # saver.save(sess,'%smodel.ckpt' % (logs_dir))
            print('Finish training.')
        except tf.errors.OutOfRangeError:
            print("Done training -- epoch limit reached.")
        finally:
            coord.request_stop()

        coord.join(threads)












