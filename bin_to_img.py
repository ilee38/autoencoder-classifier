#!/usr/bin/python

#
# Converts a binary file to a gray-scale image using numpy and Image from the
# PIL library.
#
# !!! Enter the binary file name as a command line argument (no checks for correctness) !!!
#

import numpy, os, array, sys
from PIL import Image

filename = sys.argv[1]
imgFile = filename + '.png'

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
