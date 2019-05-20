import glob
import os
import random
import numpy as np
from tensorflow.python.platform import gfile

input_data_dir = "./dataset/train/"
# 经过卷积网络得到的特征向量的存放路径
cache_dir = "./bottleneck/"

# 读取文件中的图片以及划分数据集
def create_image_dict():
    path = [x[0] for x in os.walk(input_data_dir)]
    is_root_dir = True
    for sub_dir in path:
        if is_root_dir:
            is_root_dir = False
            continue
        extension_name = ['jpg','jpeg','JPG','JPEG']
        image_list = []
        for extention in extension_name:
            file_glob = os.path.join(sub_dir,'*.' + extention)
            image_list.extend(glob.glob(file_glob))
        dir_name = os.path.basename(sub_dir)
        category = dir_name

        training_images = []
        testing_images = []
        validation_images = []

        for image_name in image_list:
            image_name = os.path.basename(image_name)
            score = np.random.randint(100)
            if score < 10:
                validation_images.append(image_name)
            elif score < 20:
                testing_images.append(image_name)
            else:
                training_images.append(image_name)
        result = {}
        result[category] = {
            "dir": dir_name,
            "training": training_images,
            "testing": testing_images,
            "validation": validation_images,
        }
        return result

#
def get_image_path(image_lists,image_dir,category,image_index,data_category):
    category_list = image_lists[category][data_category]
    actual_index = image_index % len(category_list)
    image_name = category_list[actual_index]
    sub_dir = image_lists[category]["dir"]

    full_path = os.path.join(image_dir,sub_dir,image_name)
    return full_path


# 数据的预处理，生成特征向量
def create_bottleneck(sess,image_lists,category,image_index,data_category,
                      jpeg_data_tensor,bottleneck_tensor):
    sub_dir = image_lists[category]["dir"]
    sub_dir_path = os.path.join(cache_dir,sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_image_path(image_lists,cache_dir,category,image_index,data_category) + ".txt"

    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists,input_data_dir,category,image_index,data_category)
        image_data = gfile.FastGFile(image_path,"rb").read()

        bottleneck_values = sess.run(bottleneck_tensor,feed_dict={jpeg_data_tensor:image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path,"w") as bottleneck_file:
            bottleneck_file.write(bottleneck_string)

    else:
        with open(bottleneck_path,"r") as bottleneck_file:
            bottleneck_string = bottleneck_file.read()

        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values

# 产生一个batch的特征向量和标签
def get_batch(sess,num_classes,image_lists,batch_size,data_category,
                           jpeg_data_tensor,bottleneck_tensor):
    bottlenecks = []
    labels = []
    for i in range(batch_size):
        random_index = random.randrange(num_classes)
        category = list(image_lists.keys())[random_index]

        image_index = random.randrange(65536)

        bottleneck = create_bottleneck(sess,image_lists,category,image_index,data_category,
                                              jpeg_data_tensor,bottleneck_tensor)

        label = np.zeros(num_classes,dtype=np.float32)
        label[random_index] = 1.0
        labels.append(label)
        bottlenecks.append(bottleneck)
    return bottlenecks,labels

def get_test_bottlenecks(sess,image_lists,num_classes,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks = []
    labels = []
    category_list = list(image_lists.keys())
    data_category = "testing"
    for label_index,category in enumerate(category_list):
        for image_index,unused_base_name in enumerate(image_lists[category]["testing"]):
            bottleneck = create_bottleneck(sess,image_lists,category,image_index,data_category
                                            ,jpeg_data_tensor,bottleneck_tensor)

            label = np.zeros(num_classes, dtype=np.float32)
            label[label_index] = 1.0
            labels.append(label)
            bottlenecks.append(bottleneck)
    return bottlenecks,labels










