# ** DELETE THIS **
# ** DELETE THIS **

import torch.nn as nn
from vgg_like.VGGChroClass import VGGChroClass
import warnings, math
from collections import OrderedDict

class CNNPart:
    # this class is for used in InceptionNet creation only
    # don't use this else where

    def model_build(self, chro:str, data_shape:list, first_layer:bool = True):
        # the math follows or close to one used in GoogLeNet
        chunk = VGGChroClass.chro_str2chunk(chro)
        order = OrderedDict()
        in_channel = data_shape[1]
        out_channel = int
        if first_layer:
            img_size = math.floor((data_shape[2] + data_shape[3])/2) # average height and width
        else:
            img_size = data_shape[1]
        shape_output = list(data_shape)
        ConvToLinear = False
        ConvFirst = True

        for index,x in enumerate(chunk):
            if x[0] == 'C':
                k_size = [int(x[1]), int(x[2])]
                if ConvFirst == True:
                    out_channel,pool_mul = img_size,img_size
                    ConvFirst = False
                else:
                    out_channel = pool_mul

                # we want to reserve spatial size in this class
                # pad = (kernel-1)/2 (odd)
                # pad = (kernel-1) if dilation = 2 (even)
                # prefer first kernel channel
                if k_size[0] != k_size[1]: # if not equal
                    warnings.warn("A layer isn't a square! Equaling them ..")
                    k_size[1] = k_size[0]

                pad = k_size[0] - 1

                order['conv' + str(index)] = nn.Conv2d(in_channels=in_channel,
                                                       kernel_size=k_size, out_channels=out_channel,
                                                       dilation=2, padding=pad)
                order['batch_norm' + str(index)] = nn.BatchNorm2d(out_channel)
                order['relu' + str(index)] = nn.ReLU()

                shape_output[1],in_channel = out_channel,out_channel

            elif x[0] == "P":
                k_size = (int(x[1]), int(x[2]))
                # using max tactic
                order['pool' + str(index)] = nn.MaxPool2d(kernel_size=k_size)

                # for pooling.. uh, floor and divide by k_size[0]
                p_size_h = math.floor((shape_output[2] / k_size[1]))
                p_size_w = math.floor((shape_output[3] / k_size[0]))
                shape_output[2],shape_output[3] = p_size_h,p_size_w
                pool_mul *= 2  # out_channel multi for convolution net
                in_channel = out_channel

            elif x[0] == "F":  # linear architecture
                if ConvToLinear == False:
                    #print("final shape:",linear_shape)
                    order['flat'] = nn.Flatten()
                    order['dropout'] = nn.Dropout(0.7)
                    # match the size of conv output with linear input
                    in_channel = shape_output[1] * \
                        shape_output[2]*shape_output[3]
                    ConvToLinear = True

                out_channel = int(x[1:])
                order['linear' + str(index)
                      ] = nn.Linear(in_channel, out_channel)
                order['relu' + str(index)] = nn.ReLU()

                in_channel = out_channel
            else:
                raise Exception("What is that? What the fyuck is that!?")
        # shape output pass, match match actual output size
        net = nn.Sequential(order)
        return net,shape_output