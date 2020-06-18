"""
For Regularzation
Add regularizers when creating variables or layers:

tf.layers.dense(x, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
# or
tf.get_variable('a', regularizer=tf.contrib.layers.l2_regularizer(0.001))
Add the regularization term when defining loss:

loss = ordinary_loss + tf.losses.get_regularization_loss()

"""

import json 
from collections import namedtuple
from tensorflow.keras import models, layers
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sys import exit
import tensorflow as tf
import datetime

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
    print('\nBuilding Conv Layer')
    return layers.Conv2D(layer.numberOfChannels, layer.size, activation=layer.activationFunction, input_shape=layer.inputShape)

def Max2D(layer):
    print('\nBuilding Max2D Layer')
    return layers.MaxPooling2D(layer.size)

def DropOut(layer):
    print('\nBuilding DropOut Layer')
    return layers.Dropout(float(layer.dropOutRate))

def Flatten(layer):
    print('\nBuilding Flatten Layer')
    return layers.Flatten()

def Dense(layer):
    print('\nBuilding Dense Layer')
    return layers.Dense(int(layer.size[0]), activation=layer.activationFunction)

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

# (Change to stdin)
with open('/Users/brett/TensorFlowProjects/RefineCNN/ModelsOutput.json', 'r') as file:
    data = json.load(file, object_hook=converter)
    if data is None:
        print("\n Input File Not Found")
        exit()

# Build the initial data and structure for the
# Neural Networks we will be building
(train_data, train_labels), (test_data, test_labels) \
 = mnist.load_data()

# Build training and test data
# Reshape training and test data to add an additional dimension of 1 channel
train_data = train_data.reshape((60000, 28, 28, 1))
test_data = test_data.reshape((10000, 28, 28, 1))

train_data = train_data[:6000]
test_data = test_data[:1000]

# Revise pixel data to 0.0 to 1.0, 32-bit float (this isn't quantum science)
train_data = train_data.astype('float32') / 255  # ndarray/scalar op
test_data = test_data.astype('float32') / 255

# Turn 1-value labels to 10-value vectors
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_labels = train_labels[:6000]
test_labels = test_labels[:1000]

# For Each Model in our test Mods
for m in data:
    print('\nStarting: ', m.name)
#    print(m.layer)

    # Create the NN
    nn = models.Sequential(name=m.name)

    # Dynamically add the specified layers
    for l in m.layer:
        nn.add(generateLayer(l))

    # Summarize and exit
    nn.summary()


    # Process it all, configure parameters, and get ready to train
    nn.compile(
        optimizer="rmsprop",             # Improved backprop algorithm
        loss='categorical_crossentropy', # "Misprediction" measure
        metrics=['accuracy']             # Report CCE value as we train
    )

    hst = nn.fit(train_data, train_labels, epochs = 12, batch_size = 64,
    validation_data = (test_data, test_labels))


    dataHistory = hst.history
    lCounter = 0
    for dataElement in dataHistory['val_accuracy']:
        lCounter += 1
        if dataElement > maxValAccuracy:
            maxValAccuracy = dataElement
            maxValAccModel = m.name
            bestEpoch = lCounter

# Finally, print out our best results
print('_____________________________________________')
print(' Best Results ')
print(' Winning Model:  ', maxValAccModel)
print(' Winning Epoch:  ', bestEpoch)
print(' Value Accuracy: ', maxValAccuracy)
print('_____________________________________________')
print('\n\n')

                
