import cv2
import numpy as np 
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import matplotlib.pyplot as plt 
#%matplotlib inline

train_data = 'D:/bu courses/EC601/car_test/cars_train'
test_data = 'D:/bu courses/EC601/car_test/cars_test'

def one_hot_lable(img):
	label = img.split('.')[0]
	if label == 'daisy':
		ohl = np.array([1,0])
	elif label == 'dandelion':
		ohl = np.array([0,1])
	return ohl

def train_data_with_label():
	train_images = []
	for i in tqdm(os.listdir(train_data)):
		path = os.path.join(train_data, i)
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (64,64))
		train_images.append([np.array(img), one_hot_lable(i)])
	shuffle(train_images)
	return train_images

def test_data_with_label():
	test_images = []
	for i in tqdm(os.listdir(test_data)):
		path = os.path.join(test_data, i)
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (64,64))
		test_images.append([np.array(img), one_hot_lable(i)])
	return test_images


from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

training_images = train_data_with_label()
testing_images = test_data_with_label()

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1, 64, 64, 1)
tr_lbl_data = np.array([i[1] for i in training_images])

tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1, 64, 64, 1)
tst_lbl_data = np.array([i[1] for i in testing_images]) 

model = Sequential()

model.add(InputLayer(input_shape = [64, 64, 1]))
model.add(Conv2D(filters=32, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 5, padding = 'same'))

model.add(Conv2D(filters=50, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 5, padding = 'same'))

model.add(Conv2D(filters=80, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 5, padding = 'same'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(rate = 0.5))
model.add(Dense(2, activation = 'softmax'))
optimizer = Adam(lr = 1e-3)

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x = tr_img_data, y = tr_lbl_data, epochs = 50, batch_size = 100)
model.summary()

fig = plt.figure(figsize = (14, 14))

for cnt, data in enumerate(testing_images[10: 40]):

	y = fig.add_subplot(6, 5, cnt + 1)
	img = data[0]
	data = img.reshape(1, 64, 64, 1)
	model_out = model.predict([data])

	if np.argmax(model_out) == 1:
		str_label = 'dandelion'
	else:
		str_label = 'daisy'

	y.imshow(img, cmap = 'gray')
	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False) 
plt.show()