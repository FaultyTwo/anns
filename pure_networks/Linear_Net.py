import torch.nn as nn
import warnings, math
from collections import OrderedDict

from vgg_like.VGGChroClass import VGGChroClass

class Linear_Net:
    # Create a pure Linear network from a chromosome

    @classmethod
    def model_build(self, chro:str, input_size):
        '''
        Build a simple linear network from a chromosome

        chro -> A chromosome in the string format
        input_size -> Number of features in the input

        returns a linear model
        '''
        # maybe we should seperate the chromosome checking method into its own
        # or else it will turn into "one api down, entire system down" situation

        # TODO: create a class for checking linear only
        if type(chro) is str:
            chunk = VGGChroClass.chro_str2chunk(chro)

        order = OrderedDict()
        in_channel = input_size
        out_channel = int

        for index,x in enumerate(chunk):
            if x[0] == "F":  # linear architecture
                out_channel = int(x[1:])
                order['linear' + str(index)
                      ] = nn.Linear(in_channel, out_channel)
                order['relu' + str(index)] = nn.ReLU()

                in_channel = out_channel
            else:
                raise Exception("Error: Somehow DFA didn't catch this. Send complaints too you know who.")
        # shape output pass, match match actual output size
        net = nn.Sequential(order)
        return net