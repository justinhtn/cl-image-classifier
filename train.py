import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torch import nn, optim
from torchvision import datasets
from torchvision import models as models
from torchvision import transforms
import utils

# bringing in arguments from the command line
ap = argparse.ArgumentParser(description='train.py')
ap.add_argument('data_directory', nargs='*', action='store', default='./classes/')
ap.add_argument('--save_path', action='store', default='./checkpoint.pth')
ap.add_argument('--arch', action='store', default='vgg16')
ap.add_argument('--lr', action='store', default=0.003)
ap.add_argument('--hl', action='store', default=1000)
ap.add_argument('--device', action='store', default = 'cuda')
ap.add_argument('--e', action='store',default = 5)

#The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs

pa = ap.parse_args()
where = pa.data_directory
save_path = pa.save_path
arch = pa.arch
lr = pa.lr
hlayer = pa.hl
d = pa.device
epochs = pa.e

if d == 'cpu':
    print("Using your cpu 😢. This will take awhile.")
    device = torch.device("cpu")

if d == 'cuda':
    print("Using cuda 🔥. Should be pretty quick!")
    device = torch.device("cuda:0")

# grab data
trainloader, validationloader, testloader, train_data, test_data = utils.data_load(where)

# generate model structure
model, criterion, optimizer = utils.nn_build(arch, hlayer, lr)

# train network and reassign to model
model = utils.train_pass(model, lr, epochs, criterion, optimizer, device, trainloader, validationloader)

# saves checkpoint
utils.save_model(model, arch, save_path, train_data, hlayer)

print("Training is complete! 🔥")
