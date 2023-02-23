import torch.nn as nn
import torch, warnings, math
from collections import OrderedDict

from vgg_like.VGGChroClass import VGGChroClass # for separating networks in a path
from pure_networks.CNN_Net import CNN_Net_Inception

class InceptionBlockNetwork(nn.Module):
    def __init__(self, paths:list, data_shape):
        # remember. all paths will return the sum of channels
        super().__init__()

        self.networks = OrderedDict() # store all networks until called
        self.shapes = list() # store all data_shapes for passing onto the next layer
        self.chan_sum = 0
        self.data_shape_ret = list(data_shape)

        for index,x in enumerate(paths):
            self.networks["path" + str(index)],c = CNN_Net_Inception.model_build(x, data_shape)
            self.chan_sum = self.chan_sum + int(c[1])

        # PyTorch is weird when it comes to network
        self.fake_sequence = nn.Sequential(self.networks)

        self.data_shape_ret[1] = self.chan_sum
        #print("chan_sum:",self.chan_sum)

    def forward(self,x):
        # now we need to forward it properly
        forwarder = list()
        for child in self.fake_sequence.children():
            forwarder.append(child(x))
        x_out = torch.cat(forwarder,dim=1)
        return x_out

    def get_chan_sum(self) -> int:
        # colonel, im having a mental breakdown over getting shape data from the inception network
        return self.data_shape_ret

class InceptionBlock:
    def model_build(self,chro:list,data_shape):
        pain = InceptionBlockNetwork(chro,data_shape)
        return pain, pain.get_chan_sum()