
import json 
from collections import namedtuple

# Declare the lists
lstModels = []

class LyrParams(object):
    def __init__(self, name, numberOfChannels, dropOutRate, useRegularization):
        self.name = name
        self.numberOfChannels = numberOfChannels
        self.dropOutRate = dropOutRate
        self.useRegularization = useRegularization

    """
    Custom Params Class

    def to_dict(self):
        data = {}
        data['name'] = self.name
        data['numberOfChannels'] = self.numberOfChannels
        data['dropOutRate'] = self.dropOutRate
        data['useRegularization'] = self.useRegularization

    def __init__(self, j):
        self.__dict__ = json.loads(j)
    """

class Models(object):
#    layers = []
    def __init__(self, name, layers):
        self.name = name
#        self.layers.append(layers)
#        self.layers = layers

def customModelDecoder(modelDict):
    if 'layers' in modelDict:
        return Models(modelDict['name'], modelDict['layers'])
    elif 'numberOfChannels' in modelDict:
        return LyrParams(modelDict['name'], modelDict['numberOfChannels'], modelDict['dropOutRate'], modelDict['useRegularization'])

with open('/Users/brett/TensorFlowProjects/RefineCNN/data.json', 'r') as file:
    data = json.load(file, object_hook=customModelDecoder)
#    jtopy=json.dumps(data)
#    objData = LyrParams(jtopy)
#    lstModels.append(data.Models[0])

print(data)





