"""
MNISTClassify.py - Program takes HDH5 model called "MNIST.h5"
and runs the resultant network on a test set of the data
Brett Huffman
CSCI 390 - Tpcs: Artificial Intelligence Summer Interterm 2020
Lab 1

Input:  String - the H5 file to load
Output: 
"""

import os, sys
from keras.datasets import mnist
import tensorflow as tf
import datetime
import numpy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

# Check the stdin arguments
modelfile = ''
modelNo = -1
try:
    if len(sys.argv) < 2:
        raise Exception
    modelfile = sys.argv[1]
except:
      print('\nusage: python MNISTClassify.py [ModelFile]')
      sys.exit(2)

# Build the initial data and structure for the
# Neural Networks we will be building
(train_data, train_labels), (test_data, test_labels) \
 = mnist.load_data()

#test_data = test_data[1000:2000]

# Reshape
test_data = test_data.reshape((10000, 28, 28, 1))

# Revise pixel data to 0.0 to 1.0, 32-bit float
test_data = test_data.astype('float32') / 255

# Open the file
new_model = tf.keras.models.load_model(modelfile)
# Make the prediction
prediction = new_model.predict(test_data)

# Output the rows of prediction data
for row in prediction:
    print(numpy.argmax(row))
