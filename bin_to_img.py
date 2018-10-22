#!/usr/bin/python

#
# Converts a binary file to a gray-scale image using numpy and Image from the
# PIL library.
#
# !!! Enter the binary file name as a command line argument (no checks for correctness) !!!
#

import numpy, os, array, sys
from PIL import Image

DATASET_FOLDER = './function_bins/'
#Number of "y" labels or function binaries
NUM_OF_LABELS = 30


def toImage(filename, imgFile):
    f = open(filename, 'rb')
    #length of file in bites
    ln = os.path.getsize(filename)
    WIDTH = 256
    rem = ln % WIDTH 
    #unint8 array
    a = array.array("B")
    a.fromfile(f, ln - rem)
    f.close()
    g = numpy.reshape(a, (len(a) / WIDTH, WIDTH))
    g = numpy.uint8(g)
    img = Image.fromarray(g)
    img.save(imgFile)

label_list = os.listdir(DATASET_FOLDER)
#change to the dataset's folder
os.chdir(DATASET_FOLDER)
for i in range(len(label_list)):
        os.chdir(label_list[i])
        filename = os.listdir(os.getcwd())[0]
        directory = os.getcwd() + '/'
        filepath = directory + filename
        imgFile = filename + '.png'
        toImage(filepath, imgFile)
        print imgFile + ' created'
        os.chdir('..')

