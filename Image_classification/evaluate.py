# 测试模型
import tensorflow as tf
from input_data_process import *
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def get_one_image(img_dir):
    # n = len(train)
    # ind = np.random.randint(0,n)
    # img_dir = train[ind]
    image = Image.open(img_dir)
    plt.imshow(image)
    plt.show()
    image = image.resize([224,224])
    image = np.array(image)
    return image

def evaluate_one_image():
    # test_dir = 'E:/Project/tensorflow_model/AlexNet_classification/dataset/'
    test_dir = 'E:/Project/tensorflow_model/AlexNet_classification/timg.jpg'
    # train,train_label,count = get_files(test_dir)
    image_array = get_one_image(test_dir)
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 5

        image = tf.cast(image_array,tf.float32)
        image = tf.reshape(image,[1,224,224,3])
        logit = inference(image,N_CLASSES,BATCH_SIZE)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32,shape=[224,224,3])
        logs_train_dir = 'E:/Project/tensorflow_model/AlexNet_classification/logs_vggnet/'
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Loading success, global_step is %s" % global_step)
            else:
                print("No checkpoint file found")

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print("This is a car with possibility %.6f" % prediction[:,0])
            elif max_index == 1:
                print("This is a dinosaur with possibility %.6f" % prediction[:,1])
            elif max_index == 2:
                print("This is a elephant with possibility %.6f" % prediction[:,2])
            elif max_index == 3:
                print("This is a dinosaur with possibility %.6f" % prediction[:,3])
            else:
                print("This is a dinosaur with possibility %.6f" % prediction[:,4])







