#!/usr/bin/python

#
# Implementation of a Convolutional Denoising Autoencoder (DAE) and a fully-connected
# classifier on top. Used for binary file gray-scale image reconstruction and
# classification.
#
import keras
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from keras.optimizers import RMSprop

TRAIN_DATA_FOLDER = './train_set/'
IMG_DIM = 32
NUM_CLASSES = 29

X_clean = np.load(TRAIN_DATA_FOLDER + 'orig_train_imgs.npy')
X_noisy = np.load(TRAIN_DATA_FOLDER + 'noisy_imgs_train.npy')
# The labels are the same for both clean and noisy images
Y_labels = np.load(TRAIN_DATA_FOLDER + 'train_labels.npy')

#
# Partition dataset for training and validation as follows:
# total samples = 11,020
# -----------------------
# training = 10,000 -> training set = 8,000, validation set = 2,000
# test set = 1,020
#
X_train_clean = X_clean[:8000]
Y_train_clean = Y_labels[:8000]
X_train_noisy = X_noisy[:8000]
Y_train_noisy = Y_labels[:8000]

X_val_clean = X_clean[8000:10000]
Y_val_clean = Y_labels[8000:10000]
X_val_noisy = X_noisy[8000:10000]
Y_val_noisy = Y_labels[8000:10000]

X_test_noisy = X_noisy[10000:]
X_test_clean = X_clean[10000:]
Y_test = Y_labels[10000:]

print ('Data exploration')
print ('==================')
print ('clean train set: ' + str(X_train_clean.shape))
print ('clean train labels: ' + str(Y_train_clean.shape))
print ('noisy train set: ' + str(X_train_noisy.shape))
print ('noisy train labels: ' + str(Y_train_noisy.shape))
print ('clean val set: ' + str(X_val_clean.shape))
print ('clean val labels: ' + str(Y_val_clean.shape))
print ('noisy val set: ' + str(X_val_noisy.shape))
print ('noisy val labels: ' + str(Y_val_noisy.shape))
print ('noisy test set: ' + str(X_test_noisy.shape))
print ('clean test set: ' + str(X_test_clean.shape))
print ('test labels: ' + str(Y_test.shape))
print ('==================')

"""
#plot test images
print ('Test images')
n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(X_train_clean[i].reshape(IMG_DIM, IMG_DIM))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#plot noisy images
print ('Noisy images')
n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(X_train_noisy[i].reshape(IMG_DIM, IMG_DIM))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
"""

###---------- Convolutional Denoising Autoencoder model training ------------###
input_img = Input(shape=(IMG_DIM, IMG_DIM, 1))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
encoded = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=RMSprop(), loss='mean_squared_error')

dae_train = autoencoder.fit(X_train_noisy, X_train_clean,
              epochs=50,
              batch_size=128,
              shuffle=True,
              validation_data=(X_val_noisy, X_val_clean),
              verbose=1)

"""
print ('DAE Loss')
# Plot Traininig and Validation loss
loss = dae_train.history['loss']
val_loss = dae_train.history['val_loss']
epochs = range(50)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
"""
# Predict with test data
pred = autoencoder.predict(X_test_noisy)

'''
# Plot test, noisy and reconstructed images
plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(10,20,1):
    ax = plt.subplot(2, 10, i+1)
    plt.imshow(X_test_clean[i, ..., 0], cmap='gray')
    curr_lbl = str(Y_test[i])
    plt.title(curr_lbl)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

plt.figure(figsize=(20, 4))
print("Test Images with Noise")
for i in range(10,20,1):
    bx = plt.subplot(2, 10, i+1)
    plt.imshow(X_test_noisy[i, ..., 0], cmap='gray')
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()

plt.figure(figsize=(20, 4))
print("Reconstruction of Noisy Test Images")
for i in range(10,20,1):
    cx = plt.subplot(2, 10, i+1)
    plt.imshow(pred[i, ..., 0], cmap='gray')
    cx.get_xaxis().set_visible(False)
    cx.get_yaxis().set_visible(False)
plt.show()
'''
###----- End of Convolutional Denoising Autoencoder model training ----------###

# Numpy arrays of reconstructed images. These will be used to train the final classifier
pred_train = autoencoder.predict(X_train_noisy)
pred_val = autoencoder.predict(X_val_noisy)
pred_test = autoencoder.predict(X_test_noisy)

# Make label dictionary for categorical classification
label_dict = {
    "b'netstat_'": 0,
    "b'mknod_'": 1,
    "b'openvt_'": 2,
    "b'rm_'": 3,
    "b'su_'": 4,
    "b'bash_'": 5,
    "b'rmdir_'": 6,
    "b'ip_'": 7,
    "b'cp_'": 8,
    "b'dmesg_'": 9,
    "b'dash_'": 10,
    "b'chown_'": 11,
    "b'fuser_'": 12,
    "b'init_'": 13,
    "b'kill_'": 14,
    "b'rmmod_'": 15,
    "b'insmod_'": 16,
    "b'ifconfig_'": 17,
    "b'cpio_'": 18,
    "b'mkdir_'": 19,
    "b'xtables-multi_'": 20,
    "b'ss_'": 21,
    "b'nc_'": 22,
    "b'chmod_'": 23,
    "b'dir_'": 24,
    "b'ps_'": 25,
    "b'chgrp_'": 26,
    "b'kmod_'": 27,
    "b'ls_'": 28,
}

# Create one-hot label vectors
y_train_labels = np.zeros((len(Y_train_noisy), NUM_CLASSES))
y_val_labels = np.zeros((len(Y_val_noisy), NUM_CLASSES))
for i in range(len(Y_train_noisy)):
    y_train_labels[i, label_dict[str(Y_train_noisy[i,0])]] = 1

for i in range(len(Y_val_noisy)):
    y_val_labels[i, label_dict[str(Y_val_noisy[i,0])]] = 1

print('train labels shape: ' + str(y_train_labels.shape))
print('validation labels shape: ' + str(y_val_labels.shape))

#print(Y_train_noisy[45,0])
#for n in range(NUM_CLASSES):
#    print(y_train_labels[45,n])

#print(Y_val_noisy[20,0])
#for n in range(NUM_CLASSES):
#    print(y_val_labels[20,n])


###--------- Fully-connected Classifier on top of Convolutional DAE ---------###
x = Flatten()(input_img)
x = Dense(1024, activation ='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation ='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation ='relu')(x)
x = Dropout(0.5)(x)
class_out = Dense(NUM_CLASSES, activation='softmax')(x)

classifier = Model(input_img, class_out)
classifier.summary()
classifier.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])

classifier_train = classifier.fit(pred_train, y_train_labels,
                                 epochs=50,
                                 batch_size=128,
                                 shuffle=True,
                                 validation_data=(pred_val, y_val_labels))

'''
print (' Classifier Training and Validation Loss and Accuracies')
print ('--------------------------------------------------------')
# Plot Traininig and Validation loss
loss = classifier_train.history['loss']
val_loss = classifier_train.history['val_loss']
acc = classifier_train.history['acc']
val_acc = classifier_train.history['val_acc']
epochs = range(50)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
'''
###-------------- End of Fully-connected Classifier training ----------------###