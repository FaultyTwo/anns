from vgg_like.VGG_Like_Net import VGGLikeNet, VGGLikeNSNet
from inception_like.InceptionNet import InceptionNet

# vgg-like networks
vgg_chromosome = "C22P22C22C33P22F128F10"
image_shape = [1,3,32,32] # assume we have a single sample from a CIFAR-10 dataset

vgglike = VGGLikeNet.model_build(vgg_chromosome,[1,3,32,32])
vgglikens = VGGLikeNSNet.model_build(vgg_chromosome,[1,3,32,32])

print("VggLikeNet:", vgglike)
print("VggLikeNet:", vgglikens)

# inception-like networks (W.I.P)
inception_chromosome = 'C22P22C22 ISC2216SC118SC1116I ISC2232SC1132SC1132I ISC2232SC1132SC1132I C11P22C11P22F200F10'
incept = InceptionNet.model_build(inception_chromosome,image_shape)

print("InceptionNet:", incept)