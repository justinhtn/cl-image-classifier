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
import json
import utils

torch.manual_seed(0)

# bringing in arguments from the command line
ap = argparse.ArgumentParser(description='predict.py')
ap.add_argument('--checkpoint_path', action='store', default='./checkpoint.pth')
ap.add_argument('--image_path', action='store', default= 'classes/test/1/image_06760.jpg')
ap.add_argument('--k', action='store', default=1)
ap.add_argument('--device', action='store', default = 'cuda')
ap.add_argument('--categories', action='store', default = 'cat_to_name.json')

# command line argument for test run. Comment in if test run is desired before full prediction.
# ap.add_argument('--arch', action='store', default='vgg16')

pa = ap.parse_args()
checkpoint_path = pa.checkpoint_path
image_path = pa.image_path
k_val = pa.k
d = pa.device
cat_to_name = pa.categories
# Test run argument - comment in if test run is desired.
# arch = pa.arch

# load data
trainloader, validationloader, testloader, train_data, test_data = utils.data_load()

print("Your models prediction will print below any moment 😎 ...")

# load pretrained model
model = utils.load_model('checkpoint.pth')
    
# Comment in the next line for test pass
# utils.test_pass(model, testloader, criterion, 3, device)

# load image path from command line argument
image_path = image_path

# runs the image grabbed from the command line path through our processing step
test_image = utils.process_image(image_path)

# predict based on image we recieved from last processing step
top_prob, top_labels, top_classes = utils.predict(cat_to_name, test_image, model, k_val)

# print all probabilities and flower names
print("Prediction: {}\nProbability: {}".format(top_classes, top_prob))
