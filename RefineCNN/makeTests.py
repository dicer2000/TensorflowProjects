import json
from json import JSONEncoder
from collections import namedtuple
import copy

class Layers(object):
    def __init__(self, name, layerType, size, 
        dropOutRate, numberOfChannels,
        useRegularization, activationFunction, 
        inputShape):
        self.name = name
        self.layerType = layerType
        self.size = size
        self.dropOutRate = dropOutRate
        self.numberOfChannels = numberOfChannels
        self.useRegularization = useRegularization
        self.activationFunction = activationFunction
        self.inputShape = inputShape

class Models(object):
    def __init__(self, name, layer):
        self.name = name
        self.layer = layer

class ModelEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__



models = []
layers = []

"""
def Conv2D(layer):
    print('\nBuilding Conv Layer')
    return Layers('CL', 'Conv2D', layer.size, 0.0, layer.numberOfChannels, False, layer.activationFunction,  layer.inputShape)

def Max2D(layer):
    print('\nBuilding Max2D Layer')
    return Layers('MP', 'MaxPooling2D', layer.size, 0.0, 0, False, '',  [ 0, 0, 0 ])

def DropOut(layer):
    print('\nBuilding DropOut Layer')
    return Layers('DO', 'DropOut', [0], layer.dropOutRate, 0, False, '',  [ 0, 0, 0 ])

def Flatten(layer):
    print('\nBuilding Flatten Layer')
    return Layers('FT1', 'Flatten', [0], 0.0, 32, False, '',  [ 0 ])

def Dense(layer):
    print('\nBuilding Dense Layer')
    return Layers('D11', 'Dense', layer.size, 0.0, 0, False, layer.activationFunction,  [ 0, 0, 0 ])


switcher = {
        "Conv2D": Conv2D,
        "MaxPooling2D": Max2D,
        "DropOut": DropOut,
        "Flatten": Flatten,
        "Dense": Dense
}

def ModLayers(layer, mod):
    # Get the function from switcher dictionary
    func = switcher.get(layer.layerType, "nothing")
    # Execute the function
    return func(layer)
"""
layers.clear()
layers.append(Layers('CL', 'Conv2D', [3, 3], 0.0, 32, False, 'relu',  [ 28, 28, 1 ]))
layers.append(Layers('FT', 'Flatten', [0], 0.0, 32, False, '',  [ 0 ]))
layers.append(Layers('D', 'Dense', 64, 0.0, 0, False, 'relu',  [ 0, 0, 0 ]))
layers.append(Layers('D', 'Dense', 10, 0.0, 0, False, 'softmax',  [ 0, 0, 0 ]))
models.append(Models(len(models)+1, copy.deepcopy(layers)))

layers.insert(1, Layers('MP2', 'MaxPooling2D', [2, 2], 0.0, 0, False, '',  [ 0, 0, 0 ]))
models.append(Models(len(models)+1, copy.deepcopy(layers)))

layers.insert(2, Layers('DO2', 'DropOut', [0], 0.2, 0, False, '',  [ 0, 0, 0 ]))
models.append(Models(len(models)+1, copy.deepcopy(layers)))

layers.insert(3,Layers('CL', 'Conv2D', [3, 3], 0.0, 64, False, 'relu',  [ 28, 28, 1 ]))
models.append(Models(len(models)+1, copy.deepcopy(layers)))

layers.insert(4, Layers('MP2', 'MaxPooling2D', [2, 2], 0.0, 0, False, '',  [ 0, 0, 0 ]))
models.append(Models(len(models)+1, copy.deepcopy(layers)))

layers.insert(5, Layers('DO2', 'DropOut', [0], 0.2, 0, False, '',  [ 0, 0, 0 ]))
models.append(Models(len(models)+1, copy.deepcopy(layers)))

layers.insert(6,Layers('CL', 'Conv2D', [3, 3], 0.0, 64, False, 'relu',  [ 28, 28, 1 ]))
models.append(Models(len(models)+1, copy.deepcopy(layers)))

layers.insert(7, Layers('MP2', 'MaxPooling2D', [2, 2], 0.0, 0, False, '',  [ 0, 0, 0 ]))
models.append(Models(len(models)+1, copy.deepcopy(layers)))

layers.insert(8, Layers('DO2', 'DropOut', [0], 0.2, 0, False, '',  [ 0, 0, 0 ]))
models.append(Models(len(models)+1, copy.deepcopy(layers)))

layers.insert(9, Layers('D', 'Dense', 32, 0.0, 0, False, 'relu',  [ 0, 0, 0 ]))
models.append(Models(len(models)+1, copy.deepcopy(layers)))

json_str = json.dumps(models, indent=4, cls=ModelEncoder)

print(json_str)