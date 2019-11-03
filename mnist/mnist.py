import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../mnist", one_hot=False)

X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
y_train = mnist.train.labels

X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
y_test = mnist.test.labels

train_image = open("../mnist/train_image.txt", "w")
for input in X_train:
	for data in input:
		train_image.write(str(data)+" ")

train_label = open("../mnist/train_label.txt", "w")
for data in y_train:
	train_label.write(str(data)+" ")

test_image = open("../mnist/test_image.txt", "w")
for input in X_test:
	for data in input:
		test_image.write(str(data)+" ")

test_label = open("../mnist/test_label.txt", "w")
for data in y_test:
	test_label.write(str(data)+" ")

del mnist