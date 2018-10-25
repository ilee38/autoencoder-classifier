#!/usr/bin/python

##
# Script to prepare the images to insert artificial "noise",
# thus simulating the effects of corrupted functions by the
# rootkit implementation.
#
# At the end of the process, three numpy arrays are generated (.npy files),
# which contain the dataset for training:
#
#  Filename               Description                       Array shape
# ---------------------  --------------------------------  ------------
#  orig_train_imgs.npy   original training images          (11020, 28, 28, 1)
#  noisy_imgs_train.npy  array of noisy binary images      (11020, 28, 28, 1)
#  train_labels.npy      labels vector for all samples     (11020,)
#  benign_imgs.npy       original benign binary images     (29, 28, 28, 1)
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

DATASET_FOLDER = './function_bins/'
TRAIN_DATA_FOLDER = './train_set'
NUMBER_OF_LABELS = 29   #number of benign binaries
SAMPLES_PER_BINARY = 380 #number of 'noisy' samples per each binary
DATASET_LABELS = NUMBER_OF_LABELS*SAMPLES_PER_BINARY  #total number of labels

# Array containig the sample images
X_benign_img = np.zeros((NUMBER_OF_LABELS, 28, 28))
X_train_samples = np.zeros((DATASET_LABELS, 28, 28))
# Array of labels, dtype="S15" means a string of 15 chars. numpy only accepts
# fixed-length str's as dtype
Y_train_labels = np.empty((DATASET_LABELS), dtype="S15")

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
        imgResized = img.resize((28, 28), Image.LANCZOS)
        X_benign_img[count] = np.array(imgResized)
        count += 1
        os.chdir('..')
# back to parent directory
os.chdir('..')

#
# Create "training samples". First make copies of the benign images,
# then we will add artificial noise to these copies
#
cnt = 0
for j in range(len(label_list)):
  for k in range(SAMPLES_PER_BINARY):
    X_train_samples[cnt] = X_benign_img[j]
    Y_train_labels[cnt] = label_list[j]
    cnt += 1

##
# The benign images are now in array X_benign_img[]
# And the training samples in X_train_samples[]
# Now, generate noisy images by adding gaussian noise
# to the training samples array
##

# Normalize values and convert to 28x28x1
X_train_samples = X_train_samples.astype('float32') / 255.
X_train_samples = np.reshape(X_train_samples, (len(X_train_samples), 28, 28, 1))

# Add synthetic noise: apply a gaussian noise matrix and clip the images between 0 an 1
noise_factor = 0.2
X_noisy_img = X_train_samples + noise_factor * np.random.normal(loc=0.0, scale=1.0, size = X_train_samples.shape)
X_noisy_img = np.clip(X_noisy_img, 0., 1.)

print 'train samples shape: '
print X_train_samples.shape
print 'train  noisy samples shape: '
print X_noisy_img.shape
print 'train labels shape: '
print Y_train_labels.shape

# Save the all the data in numpy arrays
# note: use np.load() to load the data when needed
os.chdir(TRAIN_DATA_FOLDER)
np.save('noisy_imgs_train.npy', X_noisy_img)
np.save('orig_train_imgs.npy', X_train_samples)
np.save('train_labels.npy', Y_train_labels)
np.save('benign_imgs.npy', X_benign_img)

# The following code was only used for testing the outputs

#index = 0
#for l in range(NUMBER_OF_LABELS):
#  print Y_train_labels[index]
#  index += 380

# Plot noisy images
#n = 10
#plt.figure(figsize=(20, 2))
#for i in range(n):
#    ax = plt.subplot(1, n, i)
#    plt.imshow(X_noisy_img[i].reshape(128, 128))
#    plt.gray()
#    plt.title(Y_train_labels[i])
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#plt.show()
