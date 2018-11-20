#!/usr/bin/python

##
# Script to generate images with data augmentation in keras,
# thus simulating the effects of corrupted functions by the
# rootkit implementation.
# Note: data augmentation should only be done with test samples
#
# At the end of the process, three numpy arrays are generated (.npy files),
# which contain the dataset for training:
#
#  Filename                   Description                   Array shape
# ---------------------     ----------------------------   ------------
# aug_clean_imgs_train.npy   augmented training images     (11020, IMG_DIMENSION, IMG_DIMENSION, 1)
#
# the training set can be partitioned to get a training and test sets
#
# Note: be sure to configure the constants DATASET_FOLDER and
# TRAIN_DATA_FOLDER per your directory structure
##

import os, glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator

DATASET_FOLDER = './function_bins/'
TRAIN_DATA_FOLDER = './train_set'
NUMBER_OF_LABELS = 29   #number of benign binaries
SAMPLES_PER_BINARY = 380 #number of 'noisy' samples per each binary
DATASET_LABELS = NUMBER_OF_LABELS*SAMPLES_PER_BINARY  #total number of labels
IMG_DIMENSION = 32  #image's width and height dimensions

# Array containig the sample images
X_benign_img = np.zeros((NUMBER_OF_LABELS, IMG_DIMENSION, IMG_DIMENSION))
X_train_samples = np.zeros((DATASET_LABELS, IMG_DIMENSION, IMG_DIMENSION))
X_train_aug_clean = np.zeros((IMG_DIMENSION, IMG_DIMENSION, 1))
# Array of labels, dtype="S15" means a string of 15 chars. numpy only accepts
# fixed-length str's as dtype
Y_train_labels = np.empty((DATASET_LABELS, 1), dtype="S15")

# List of dataset's subdirectories, each subdir is a binary file
# function (e.g. a label of our data set)
label_list = os.listdir(DATASET_FOLDER)
label_list.remove(".DS_Store")  #for MacOS
# Change to the dataset's folder
os.chdir(DATASET_FOLDER)
count = 0
for i in range(len(label_list)):
        #enter directory containing the .png file
        os.chdir(label_list[i])
        filename = glob.glob('*.png')[0]
        img = Image.open(filename)
        imgResized = img.resize((IMG_DIMENSION, IMG_DIMENSION), Image.LANCZOS)
        X_benign_img[count] = np.array(imgResized)
        count += 1
        os.chdir('..')
# back to parent directory
os.chdir('..')

#
# Create "training samples". First make copies of the benign images,
# then we will augment these copies
#
cnt = 0
for k in range(SAMPLES_PER_BINARY):
  for j in range(len(label_list)):
    X_train_samples[cnt] = X_benign_img[j]
    Y_train_labels[cnt] = label_list[j]
    cnt += 1


##
# The benign images are now in array X_benign_img[] (29,IMG_DIMENSION,IMG_DIMENSION)
# And the training samples in X_train_samples[] (11020, IMG_DIMENSION, IMG_DIMENSION)
##

# Normalize values and convert to IMG_DIMENSIONxIMG_DIMENSIONx1
X_train_samples = X_train_samples.astype('float32') / 255.
X_train_samples = np.reshape(X_train_samples, (len(X_train_samples), IMG_DIMENSION, IMG_DIMENSION, 1))

X_train_aug_clean = X_train_samples

# Create object for data ugmentation in keras
datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

for img in range(len(X_train_aug_clean)):
  X_train_aug_clean[img] = datagen.random_transform(X_train_aug_clean[img])

print 'augmented samples shape: ' + str(X_train_aug_clean.shape)
print 'augmented labels shape: ' + str(Y_train_labels.shape)

# Save the all the data in numpy arrays
# note: use np.load() to load the data when needed
os.chdir(TRAIN_DATA_FOLDER)
np.save('aug_clean_imgs_train.npy', X_train_aug_clean)
np.save('aug_train_labels.npy', Y_train_labels)

"""
### The following code was only used for testing ###
for l in range(35):
  print Y_train_labels[l]


#Plot noisy images
n = 35
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i)
    plt.imshow(X_train_aug_clean[i].reshape(IMG_DIMENSION, IMG_DIMENSION))
    plt.gray()
    plt.title(Y_train_labels[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
"""