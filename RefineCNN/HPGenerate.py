"""
HPGenerate.py - Program takes a model scenerio and epochs 
input, generates the model based on those hyperparameters, 
and saves it to and HDH5 model called "MNIST.h5"
Brett Huffman
CSCI 390 - Tpcs: Artificial Intelligence Summer Interterm 2020
Lab 1

Input:  String - Input file name (json models file)
        Int - Model No
        Int - Epoch
Output: An MNIST.h5 file
"""

import json 
from collections import namedtuple
from tensorflow.keras import models, layers
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sys import exit
import sys
import tensorflow as tf
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Check the stdin arguments
inputfile = ''
modelNo = 1
epochNo = 1
try:
    if len(sys.argv) < 4:
        raise Exception
    inputfile = sys.argv[1]
    modelNo = int(sys.argv[2])
    epochNo = int(sys.argv[3])
except:
      print('\nusage: python HPGenerate.py [InputFile] Model_Number Epoch_Number')
      sys.exit(2)

# The whole reason we are doing this:
# Accuracy Determination
maxValAccuracy = 0.0
maxValAccModel = 0
bestEpoch = 0

#   Build the list of Models to generate the
#   test suite (comes from the stdin json file)
#   Mod = The test models to generate
#   Layer = The layers within each model
#   'data' will hold all the model data
Mods = namedtuple('Model', ['name', 'layer'])
Lays = namedtuple('Layer', ['name', 'layerType', 'size', 'dropOutRate', 'numberOfChannels', 'useRegularization', 'activationFunction', 'inputShape'])

def converter(dict):
    if 'layer' in dict:
        return Mods(**dict)
    elif 'numberOfChannels' in dict:
        return Lays(**dict)

def Conv2D(layer):
    print('Conv2D Layer: Channels: ', layer.numberOfChannels,' Act: ',layer.activationFunction, ' Shape: ', layer.inputShape, ' ', layer.size)
    return layers.Conv2D(layer.numberOfChannels, layer.size, 
        activation=layer.activationFunction, input_shape=layer.inputShape, 
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        activity_regularizer=tf.keras.regularizers.l2(0.01))

def Max2D(layer):
    print('Max2D Layer: ',layer.size)
    return layers.MaxPooling2D(layer.size)

def DropOut(layer):
    print('Dropout Layer: ', layer.dropOutRate)
    return layers.Dropout(float(layer.dropOutRate))

def Flatten(layer):
    print('Flatten Layer')
    return layers.Flatten()

def Dense(layer):
    print('Dense Layer: Act: ', layer.activationFunction, ' ', layer.size[0])
    return layers.Dense(int(layer.size[0]), 
        activation=layer.activationFunction, 
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        activity_regularizer=tf.keras.regularizers.l2(0.01))

switcher = {
        "Conv2D": Conv2D,
        "MaxPooling2D": Max2D,
        "DropOut": DropOut,
        "Flatten": Flatten,
        "Dense": Dense
}

def generateLayer(layer):
    # Get the function from switcher dictionary
    func = switcher.get(layer.layerType, "nothing")
    # Execute the function
    return func(layer)

# Using the file specified in command line, open json file
with open(inputfile, 'r') as file:
    data = json.load(file, object_hook=converter)
    if data is None:
        print("\n Input File Not Found")
        exit()

# Build the initial data and structure for the
# Neural Networks we will be building
(train_data, train_labels), (test_data, test_labels) \
 = mnist.load_data()

# Augement the data
datagen = ImageDataGenerator(
   rotation_range=5,
   shear_range=.2,
   zoom_range=.1,
   width_shift_range=5,
   height_shift_range=2
)

# Build training and test data
# Reshape training and test data to add an additional dimension of 1 channel
train_data = train_data.reshape((60000, 28, 28, 1))
test_data = test_data.reshape((10000, 28, 28, 1))

train_data = train_data[:10000]
test_data = test_data[:1000]

# Revise pixel data to 0.0 to 1.0, 32-bit float (this isn't quantum science)
train_data = train_data.astype('float32') / 255  # ndarray/scalar op
test_data = test_data.astype('float32') / 255

# Turn 1-value labels to 10-value vectors
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_labels = train_labels[:10000]
test_labels = test_labels[:1000]

# For Each Model in our test Mods
for m in data:

    # Only process the one model specified
    if modelNo > 0 and m.name != modelNo:
        continue

    print('_____________________________________________')
    print('Model: ', m.name)
#    print(m.layer)

    # Create the NN
    nn = models.Sequential(name=str(m.name))

    # Dynamically add the specified layers
    for l in m.layer:
        nn.add(generateLayer(l))

    # Summarize
    #nn.summary()


    # Process it all, configure parameters, and get ready to train
    nn.compile(
        optimizer="RMSprop",             # Improved backprop algorithm
        loss='categorical_crossentropy', # "Misprediction" measure
        metrics=['accuracy']             # Report CCE value as we train
    )

    # For when using Augmentation Data
    train_generator = datagen.flow(train_data, train_labels, 
        batch_size = 128, shuffle = True)
    hst = nn.fit(train_generator, epochs = epochNo,
        validation_data = (test_data, test_labels),
        verbose=1)

    # For when using no Augmentation Data
#    hst = nn.fit(train_data, train_labels, epochs = 12, batch_size = 64,
#        validation_data = (test_data, test_labels),
#        verbose=0)

    # Save H5 file now
    nn.save('./MNIST.h5') 

    # Finally, print out our best results
    print('_____________________________________________')
    print(' Save Complete')
    print('_____________________________________________')
    print('\n\n')
    exit()

                
