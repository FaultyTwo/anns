from AutoNN import AutoNN
from vgg_like.VGGChroClass import VGGChroClass
from vgg_like.VGGGeneOperator import VGGGeneOperator
from vgg_like.VGG_Like_Net import VGGLikeNet
import torch,torchvision
from logger import Logger
import time, random

batch_size_train = 128
batch_size_test = 1000
random_seed = 1
torch.manual_seed(random_seed)
random.seed(random_seed)

if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

train_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.CIFAR10('data/', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
                            batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.CIFAR10('data/', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
                            batch_size=batch_size_test, shuffle=True)

print("training batch size:",batch_size_train)
print("testing batch size:",batch_size_test)

control_parents = ['SC22C33F20F10T','SC33P22F50F10T','SC33C22P22F10T','SC33F100F50F10T',
'SC33P22F50F10T','SC22C22F100F10T','SC33P22P22F10T','SC55F50F50F10T',
'SC44C22F30F100F10T','SC33P22C33P22F10T','SC33C44C22F50F10T','SC44C44P22F30F10T',
'SC33P22C33P22F50F10T','SC33C33C33P22F100F10T','SC22F10F20F40F10T','SC33P33F50F20F10F10T']

random_parents = VGGChroClass.chro_generate(16,(4,6),10)

print("Timer is running ...")
tic = time.perf_counter()

n = AutoNN()
# this is a big yike moments
final = n.GeneticAlgorithm(32,random_parents,train_loader,test_loader,
VGGLikeNet, VGGGeneOperator.onep_crossover, VGGGeneOperator.uniform_mutation,
32,0.001,256,device=device,plot_name="ADAM_BATCH_SPATIAL_1_32_32.png")

print("\n\nTimer is stopping ...")
toc = time.perf_counter()
print("Elapse:", round(toc - tic,4))

print("Best candidates:", final)
print("Fittest member:", final[0])
print("Evaluation successfully .. quiting ..")
