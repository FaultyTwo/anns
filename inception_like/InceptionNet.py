# Well .. now what?

from .CNNPart import CNNPart
from .InceptionChroClass import InceptionChroClass
from .InceptionBlock import InceptionBlock
from collections import OrderedDict
import torch.nn as nn

from pure_networks.CNN_Net import CNN_Net

class InceptionNet:

    def model_build(self, chunk:list, data_shape:list):
        # Chromosome reminder
        # There are two parts in Inception
        # Basic CNN, and that spider-web part
        # for this. it should be turned into chunks like this
        # Ex. ["C22C22","I-SC22-8-C33-16-SC33-16-C22-32-SC11-32-I"] * exclude hyphens
        # Or split and process to look like
        # Ex. ["C22C22".["C22","C22","C22"],"F10"]
        # where string is basic layer-based, and list is inception section
        # so first. we need to separate cnn + inception part first.

        if type(chunk) is str:
            chunk = InceptionChroClass.chro_str2chunk(chunk)
        elif type(chunk) is list:
            pass
        else:
            raise TypeError
        
        #print(chunk)

        models = list() # to build later
        order = OrderedDict()
        incept = InceptionBlock()
        shape_output = list(data_shape)
        
        for index,x in enumerate(chunk):
            # TODO: linear section for the last layer
            # match the output shape too, yas
            if type(x) is str: # is simple cnn
                # mutating this layer shouldn't include 'F' for obvious reason
                boo = True if index == 0 else False
                cnn_model,shape_output = CNN_Net.model_build(x,shape_output,boo,True)
                order['cnnpart' + str(index)] = cnn_model
            elif type(x) is list: # is inception
                # splitted into chunk nicely
                # now.. to create inception block
                incept_model,shape_output = incept.model_build(x,shape_output)
                order['inception' + str(index)] = incept_model
            else:
                raise Exception("Coca cola expresso")
        
        order['softmax'] = nn.LogSoftmax(dim=1) # good ol' log softmax
        net = nn.Sequential(order)
        return net