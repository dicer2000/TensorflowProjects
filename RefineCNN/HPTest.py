"""
HPTest.py - Program to test multiple versions of a CNN Models
against the Mnist dataset.  Program will determine best model
and epoch combination based on the Value Accuracy returned
for the average of three runs through the data.
Brett Huffman
CSCI 390 - Tpcs: Artificial Intelligence Summer Interterm 2020
Lab 1

Input:  File from which to build CNN models in JSON format
        (optionally) Model_No - directs the program to build 
        the Model and output it to a MNINST.h5 file
Output: Text reporting each model's layer configurations and
        epoch accuracy
        (optionally) MNINST.h5
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

# Global Vars
MAX_EPOCH = 12

# Vars for tracking totals
OverallAvg_MaxAcc = 0.0
OverallBestEpoch = 0

# Check the stdin arguments
inputfile = ''
modelNo = -1
try:
    if len(sys.argv) < 2:
        raise Exception
    inputfile = sys.argv[1]
    if len(sys.argv) > 2:
        modelNo = int(sys.argv[2])
except:
      print('\nusage: HPTest.py [InputFile] <Model_Number>')
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
Mods = namedtuple('Model', ['name', 'layer', 'bestResults', 'bestResultEpoch'])
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

# Repeat each Model test 3 times to find our best avg results
for Modi in [0, 1, 2]:
    # For Each Model in our test Mods
    for m in data:

        # temporary
        if m.name > 3:
            continue


        # Only process one model if it's specified in ARGV
        if modelNo > 0 and m.name != modelNo:
            continue

        print('_____________________________________________')
        print('Model: ', m.name)
    #    print(m.layer)

        # Setup an early-stopping callback so that once we have 
        # two decreasing val_accuracy values, stop processing that model
        earlyStopCallback = tf.keras.callbacks.EarlyStopping(mode='auto', 
            monitor='val_accuracy', min_delta=0.0001, patience=1)
        
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
        hst = nn.fit(train_generator, epochs = MAX_EPOCH,
            validation_data = (test_data, test_labels),
            verbose=1, callbacks=[earlyStopCallback])

        # For when using no Augmentation Data
    #    hst = nn.fit(train_data, train_labels, epochs = 12, batch_size = 64,
    #        validation_data = (test_data, test_labels),
    #        verbose=0)

        # If we specified to save this model, do it now
        if modelNo > -1:
            nn.save('./MNIST.model')
            print('Model saved to file')

        # Save the history
        dataHistory = hst.history
        lCounter = 0

        # Find the winner of this model run
        # and print it
        modelMaxAcc = max(dataHistory['val_accuracy'])
        modelBestEpoch = [i for i, j in enumerate(dataHistory['val_accuracy']) if j == modelMaxAcc]
        print(' Best Model Epoch: ', modelBestEpoch[0])
        print(' Best Model Val Accuracy: ', modelMaxAcc)

        # Save the winner to an array of results in this model object
        m.bestResults[Modi]=modelMaxAcc
        m.bestResultEpoch[Modi]=modelBestEpoch[0]

    """
    # Look for an All-Time winner based on the average of the
    # three results given back and store it for later use
    for dataElement in dataHistory['val_accuracy']:
        lCounter += 1
        # Save the best Model & Epoch if it exceeds the
        # Max Val Accuracy for all our prior tests
        if dataElement > maxValAccuracy:
            maxValAccuracy = dataElement
            maxValAccModel = m.name
            bestEpoch = lCounter
"""

# Define and average function
def Average(lst): 
    return sum(lst) / len(lst)

# Finally, Look for an All-Time winner based on the 
# average of the three results given back and
# print out that best average with best epoch
bestModel = 0
bestEpoch = 0
bestAvgValAccuracy = 0.0
for m in data:
    print(m.bestResults)
    lModelAvg = Average(m.bestResults)
    if(lModelAvg > bestAvgValAccuracy):
        # Store the best average and the epoch
        # for the best result
        bestModel = m.name
        OverallAvg_MaxAcc = lModelAvg
        modelMaxAcc = max(m.bestResults)
        OverallBestEpoch = [i for i, j in enumerate(m.bestResults) if j == modelMaxAcc]

print('_____________________________________________')
print(' Best Results ')
print(' Winning Model:  ', bestModel)
print(' Winning Epoch:  ', OverallBestEpoch[0]+1)
print(' AVG Value Accuracy: ', OverallAvg_MaxAcc)
print('_____________________________________________')
print('\n\n')