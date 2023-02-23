from .VGGChroClass import VGGChroClass
from pure_networks.CNN_Net import CNN_Net
import torch.nn as nn

class VGGLikeNet:
    '''
    Build a VGG-like model from a chromosome.
    '''
    @classmethod
    def model_build(self, chunk:str, data_shape:list):

        if VGGChroClass.chro_check(chunk) == -1:
            raise Exception("Bad Chromosome:", chunk)
        
        model,_ = CNN_Net.model_build(chunk, data_shape, keep_spatial = True)
        model.add_module("softmax",nn.Softmax(dim=1))

        return model
    
class VGGLikeNSNet:
    '''
    Build a VGG-like model from a chromosome, but doesn't keep spatial dimensions when convoluting.
    '''

    @classmethod
    def model_build(self, chunk:list, data_shape:list):

        if VGGChroClass.chro_check(chunk) == -1:
            raise Exception("Bad Chromosome:", chunk)
        
        model,_ = CNN_Net.model_build(chunk, data_shape, keep_spatial = False)
        model.add_module("softmax",nn.Softmax(dim=1))

        return model