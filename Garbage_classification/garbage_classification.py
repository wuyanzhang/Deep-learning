import numpy as np  
import os
import keras
import cv2 
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.models import Sequential
from sklearn.utils import shuffle

train_dir = "./dataset/train"
test_dir = "./dataset/test"
classes = ['dry','wet','hazardous','recycle']

def load_data(directory):
	size = (150,150)
	images = []
	labels = []
	for dir in os.listdir(directory):
		print("Loading data from",dir,":")
		for file in os.listdir(dir):
			img_path = directory + "/" + dir + "/" + dir
			img = cv2.imread(img_path)
			img = cv2.resize(img,size)
			img.append(img)
			if dir == classes[0]:
				label = 0
			elif dir == classes[1]:
				label = 1
			elif dir == classes[2]:
				label = 2
			elif dir == classes[3]:
				label = 3
			labels.append(label)
		print("Finished")

	images, labels = shuffle(images, labels)
	images = np.array(images)
    images = images.astype('float32')/255.0
    labels = np.array(labels)
    # Convert labels to categorical one-hot encoding
    labels = keras.utils.to_categorical(labels, n_classes)

	return images,labels

model = Sequential()

# Dense表示全连接层
model.add(Conv2D(32, kernel_size =[5,5], strides = 2, activation = 'relu', input_shape = (150,150,3)))
model.add(MaxPool2D(pool_size = [2,2], strides = 2))
model.add(Conv2D(64, kernel_size = [3,3], padding = 'same', activation = "relu"))
model.add(Conv2D(64, kernel_size = [3,3], padding = 'same', activation = "relu"))
model.add(MaxPool2D(pool_size = [2,2], strides = 2))
model.add(Conv2D(128, kernel_size = [3,3], activation = "relu"))
model.add(Conv2D(128, kernel_size = [3,3], activation = "relu"))
model.add(MaxPool2D(pool_size = [2,2], strides = 2))
model.add(Conv2D(256, kernel_size = [3,3], activation = "relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dense(n_classes, activation = 'softmax'))

# 输出模型各层的参数状况
model.summary()

X_train, Y_train = load_data(train_dir)

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
# Train the model, iterating on the data in batches of 32 samples
history = model.fit(X_train,Y_train,epochs=10,batch_size=32)
# fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况
print(history.history['acc'])

X_test, Y_test = load_data(test_dir)
# 返回的是损失值和你选定的指标值（例如，精度accuracy）
metrics = model.evaluate(X_test,Y_test)
print("Model metrics:",model.metrics_names)
print("Testing Accuracy:",metrics[1])