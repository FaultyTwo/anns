import torch.nn as nn
import warnings, math
from collections import OrderedDict

from vgg_like.VGGChroClass import VGGChroClass

class CNN_Net:
    # Create a pure CNN from a chromosome

    @classmethod
    def model_build(self, chro:str, data_shape:list, 
                    first_layer:bool = True, keep_spatial = False
                    , batch_norm:bool = True):
        '''
        chro -> A chromosome in the string format
        accept chunks too
        data_shape -> Shape of dataset (image)
        first_layer -> True if this is the first chunk in the network
        * For pure network, leave this to True
        * For any networks that requires seperation, set this to False
        keep_spatial -> True if you want to keep spatial dimension
        batch_norm -> Apply batch normalizer or not

        returns a tuple of (model, input_shape)
        '''
        
        if type(chro) is str:
            chunk = VGGChroClass.chro_str2chunk(chro)
            
        order = OrderedDict()
        in_channel = data_shape[1]
        out_channel = int
        # print the sentence
        
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

                pad = k_size[0] - 1 if keep_spatial else 0
                dil = 2 if keep_spatial else 1

                order['conv' + str(index)] = nn.Conv2d(in_channels=in_channel,
                                                       kernel_size=k_size, out_channels=out_channel,
                                                       dilation=dil, padding=pad)
                if batch_norm:
                    order['batch_norm' + str(index)] = nn.BatchNorm2d(out_channel)
                order['relu' + str(index)] = nn.ReLU()

                if keep_spatial == False:
                    c_size_h = (shape_output[2] - k_size[0]) + 1
                    c_size_w = (shape_output[3] - k_size[1]) + 1
                    shape_output[2],shape_output[3] = c_size_h,c_size_w

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
                    order['dropout'] = nn.Dropout(0.5)# no brainer
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
                raise Exception("Error: Somehow DFA didn't catch this. Send complaints too you know who.")
        # shape output pass, match match actual output size
        net = nn.Sequential(order)
        return net,shape_output
    
class CNN_Net_Inception:
    # INCEPTION SUBPATHS
    # USE WITH CARES

    @classmethod
    def model_build(self, chro:str, data_shape:list):
        
        # subpaths don't have linear layer
        chunk = VGGChroClass.chro_str2chunk(chro)
        order = OrderedDict()
        in_channel = data_shape[1]
        out_channel = int
        shape_output = list(data_shape)

        # no more calculating channels
        # inception is manually targeted

        for index,x in enumerate(chunk):
            if x[0] == 'C':
                k_size = [int(x[1]), int(x[2])]
                out_channel = int(x[3:])
                # reserve spatial
                if k_size[0] != k_size[1]: # if not equal
                    warnings.warn("A layer isn't a square! Equaling them ..")
                    k_size[1] = k_size[0]
                
                pad, dilation = 0,1
                if k_size[0] % 2 != 0: # even
                    pad = math.floor((k_size[0] - 1)/2)
                else: # dilation = 2
                    pad = k_size[0] - 1
                    dilation = 2

                order['conv' + str(index)] = nn.Conv2d(in_channels=in_channel,
                                                       kernel_size=k_size, out_channels=out_channel,
                                                       dilation=dilation, padding=pad)
                order['batch_norm' + str(index)] = nn.BatchNorm2d(out_channel)
                order['relu' + str(index)] = nn.ReLU()

                shape_output[1],in_channel = out_channel,out_channel

            elif x[0] == "P":
                # we have problem
                # for inception to work, we need to keep the spatial size of pooling
                # i mean .. eh?
                k_size = (int(x[1]), int(x[2]))
                order['pool' + str(index)] = nn.MaxPool2d(kernel_size=k_size)

                in_channel = out_channel
            else:
                raise Exception("Error: Somehow DFA didn't catch this. Send complaints too you know who.")
        # shape output pass, match match actual output size
        net = nn.Sequential(order)
        return net,shape_output