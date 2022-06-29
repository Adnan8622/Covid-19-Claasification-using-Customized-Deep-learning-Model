# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:35:52 2020

@author: DeepLearning
"""

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
import keras 
from keras.models import Sequential
from keras import layers
from keras import models
from keras.layers import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D, SeparableConv2D
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os
path='D:/Adnan Hussain/Wild Animal Project/Covid work/Covid_Dataset'

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 30
INIT_LR =0.0001 #1e-3
BS = 32
IMAGE_DIMS = (224, 224, 3)

# grab the image paths and randomly shuffle them
print("loading images...")
imagePaths = sorted(list(paths.list_images(path)))
random.seed(42)
random.shuffle(imagePaths)
# initialize the data and labels
data = []
labels = []

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)

	# extract set of class labels from the image path and update the
	# labels list
	l = label = imagePath.split(os.path.sep)[-2].split("_")
	labels.append(l)
	
print(labels)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
print(labels)

# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))


# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.30, random_state=42)
print("Train X = ",trainX.shape)
print("Test X = ",testX.shape)
print("Train Y = ",trainY.shape)
print("Test Y = ",testY.shape)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
	
model = Sequential([
Conv2D(64, (3, 3), input_shape=IMAGE_DIMS, padding='same', activation='relu'),
Conv2D(64, (3, 3), activation='relu', padding='same'),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(128, (3, 3), activation='relu', padding='same'),
Conv2D(128, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(256, (3, 3), activation='relu', padding='same',),
Conv2D(256, (3, 3), activation='relu', padding='same',),
Conv2D(256, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Flatten(),
Dense(4096, activation='relu'),
Dropout(3),
Dense(4096, activation='relu'),
Dense(3, activation='softmax')
])

# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print(" training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)
model.save('COVID19-Model-alexnet11.h5')
import matplotlib.pyplot as plt
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig('Covid_plot-alexnet11.png')
plt.show()


from sklearn.metrics import classification_report, confusion_matrix
num_of_test_samples = 1034
batch_size=32
target_names = ["Covid19","Normal","PNEUMONIA"] 
#Confution Matrix and Classification Report
Y_pred = model.predict(testX, num_of_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
testY = np.argmax(testY, axis=1)
print('Confusion Matrix',y_pred)
cm = confusion_matrix(testY, y_pred)
print(cm)
print('Classification Report')
print(classification_report(testY, y_pred, target_names=target_names))

import matplotlib.pyplot as plt
import matplotlib.pyplot 
from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=False,
                                class_names=target_names)

plt.savefig("Confusion Matrix alexnet11")
plt.show()

import seaborn as sn
import pandas as pd
plt.figure(figsize=(20,20))
sn.set(font_scale=2.4) # for label size
sn.heatmap(cm, annot=True, annot_kws={"size":20}) # font size
plt.savefig("covid19 Confusion Matrix11 ")
plt.show()