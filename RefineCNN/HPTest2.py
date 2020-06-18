
import json 
from collections import namedtuple

Model = namedtuple('Model', ['name', 'layer'])
Layer = namedtuple('Layer', ['name', 'dropOutRate', 'numberOfChannels', 'useRegularization', 'activationFunction', 'inputShape'])

def converter(dict):
    if 'layer' in dict:
        return Model(**dict)
    elif 'numberOfChannels' in dict:
        return Layer(**dict)

with open('/Users/brett/TensorFlowProjects/RefineCNN/data.json', 'r') as file:
    data = json.load(file, object_hook=converter)
    print(data)







