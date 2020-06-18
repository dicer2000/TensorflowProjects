import json
from json import JSONEncoder
from collections import namedtuple
import copy

def string2tuple(input):
    val = type(input)
    return tuple(map(int, input.split(',')))



class Layers(object):
    def __init__(self, name, layerType, size, 
        dropOutRate, numberOfChannels,
        useRegularization, activationFunction, 
        inputShape):
        self.name = name
        self.layerType = layerType
        self.size = string2tuple(size)
        self.dropOutRate = dropOutRate
        self.numberOfChannels = numberOfChannels
        self.useRegularization = useRegularization
        self.activationFunction = activationFunction
        self.inputShape = string2tuple(inputShape)

class Models(object):
    def __init__(self, name, layer):
        self.name = name
        self.layer = layer

class ModelEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__


models = []
layers = []

# Replace all quotes in input file
with open('/Users/brett/TensorFlowProjects/RefineCNN/Models1.txt', 'r') as f, open('/Users/brett/TensorFlowProjects/RefineCNN/ModelsOutput.txt', 'w') as fo:
    for line in f:
        fo.write(line.replace('"', '').replace("'", ""))

with open('/Users/brett/TensorFlowProjects/RefineCNN/ModelsOutput.txt','r') as f:
    for line in f:
        tokenPointer = 1
        tokens = line.split()
        layers.clear()

        # Go through all tokens in line
        while tokenPointer < len(tokens):
            # Determine Layer type
            if tokens[tokenPointer] == 'C':
                layers.append(Layers('CL', 'Conv2D', tokens[tokenPointer+1], 0.0, tokens[tokenPointer+2], False, tokens[tokenPointer+3], tokens[tokenPointer+4]))
                tokenPointer+=5
            elif tokens[tokenPointer] == 'F':
                layers.append(Layers('FT', 'Flatten', "0", 0.0, 32, False, '', "0"))
                tokenPointer+=1
            elif tokens[tokenPointer] == 'D':
                layers.append(Layers('D', 'Dense', tokens[tokenPointer+1], 0.0, 0, False, tokens[tokenPointer+2],  "0,0,0"))
                tokenPointer+=3
            elif tokens[tokenPointer] == 'M':
                layers.append(Layers('MP2', 'MaxPooling2D', tokens[tokenPointer+1], 0.0, 0, False, '',  "0,0,0"))
                tokenPointer+=2
            elif tokens[tokenPointer] == 'R':
                layers.append(Layers('DO2', 'DropOut', "0", tokens[tokenPointer+1], 0, False, '',  "0,0,0"))
                tokenPointer+=2

        # Add the model at the end of each line
        models.append(Models(len(models)+1, copy.deepcopy(layers)))

# dump all the models to JSON in stdout
json_str = json.dumps(models, indent=4, cls=ModelEncoder)
print(json_str)
