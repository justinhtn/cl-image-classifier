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
import copy
import time
from tqdm import tqdm

def data_load(where = './classes'):
    '''
    Takes in path to data, returns loaders for training, validation
    and test sets.
    '''
     
    data_directory = where
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'
    
    # building out the transfomers for each segment of our data

    training_transforms = transforms.Compose([transforms.RandomRotation(10),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # pulling out the data, transfoming it and assigning it to the right vaiables
    train_data = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=testing_transforms)

    # building out the data loader and setting the batches and shuffles
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32) 
    
    return trainloader, validationloader, testloader, train_data, test_data

def nn_build(arch, hlayer, lr):
    '''
    This is soley a reusable nn architecture function. Rather than take in long params,
    this provides a easy way to edit the necessary variables and returns what just what you need.
    '''
    # loading in the pretrained model  
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet':
        model = models.densenet161()

    # freezing the params of the already trained model
    for param in model.parameters():
        param.requires_grad = False

    # we like the features the pretrained model brings, but the output isn't built in a way to
    # accomodate our problem so we need to build another forward classifier for the job.
    model.classifier = nn.Sequential(nn.Dropout(0.3),
                                     nn.Linear(25088, hlayer),
                                     nn.ReLU(),
                                     nn.Linear(hlayer, 102),
                                     nn.LogSoftmax(dim=1))
    
    # define our loss functions and optmizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    return model, criterion, optimizer

def train_pass(model, lr, epochs, criterion, optimizer, device, trainloader, validationloader): 
    '''This should do a training and evaluation pass using the data from the
    training data loader and the validation data loader. If all goes according to plan
    I should see the training loss go down while training and see that the evaluation
    accuracy is high. We should expect a validation accuracy of above 70%.
    '''
    
    for e in range(epochs):
        
        best_model_wts = copy.deepcopy(model.state_dict())
        best_accuracy = 0
        
        # setting training loss, validation loss and validation accuracy counters at zero
        train_loss = 0
        val_loss = 0
        val_accuracy = 0
        
        # loading images and labels from training set
        print("Epoch {}/{}...".format(e+1, epochs))
        for inputs, labels in tqdm(trainloader):
            # send them to cuda device
            inputs, labels = inputs.to(device), labels.to(device)
            # zero out the gradients on each pass 
            optimizer.zero_grad()
            # sending model to cuda if available
            model.to(device)
            # get the ouput of the model
            output = model(inputs)
            # grab the loss
            loss = criterion(output, labels)
            # get new gradients
            loss.backward()
            # do a gradient descent step with new gradients
            optimizer.step()
            # update the running_loss based on the loss value for each iteration
            train_loss += loss.item()

        # get model into eval mode
        model.eval()

        # turning off updating gradients for test run
        with torch.no_grad():
            # grabbing inputs and labels of test run
            for inputs, labels in validationloader:
                # sending to cuda
                inputs, labels = inputs.to(device), labels.to(device)
                # get output of model
                output = model(inputs)
                # getting loss 
                val_loss = criterion(output, labels)
                # update the test loss with new loss on each run
                val_loss += val_loss.item()

                # getting accuracy AKA doing some magic I don't quite understand yet
                # getting the probablity outputs 
                probs = torch.exp(output)
                top_prob, top_class = probs.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                if val_accuracy > best_accuracy:
                    best_acc = val_accuracy
                    best_model_wts = copy.deepcopy(model.state_dict())
                
        # put model back in training mode for next iteration          
        model.train()  

        print(f"Training Loss: {train_loss/len(trainloader):.3f}... "
              f"Validation Loss: {val_loss/len(validationloader):.3f}... "
              f"Validation accuracy: {val_accuracy/len(validationloader):.3f}... ") 
            # load best model weights
    model.load_state_dict(best_model_wts)
    return model
       
def save_model(model, arch, save_path, train_data, hlayer, lr):
    model.class_to_idx = train_data.class_to_idx
    torch.save({
        'arch': arch,
        'hlayer':hlayer,
        'lr':lr,
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx},
        save_path)
 
def load_model(checkpoint_path):
    '''This function should take the checkpoint file path, load the checkpoint and
    extract the model information and return that fantastic model that was created earlier'''
    
    # use cpu to load
    device = torch.device('cpu')
    # loading in the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # pulling in optional hidden layer if different from default
    hlayer = checkpoint['hlayer']
    
    # pulling in architecture
    arch = checkpoint['arch']
    
    # building base model
    model,_,_ = nn_build(arch, hlayer)

    # grab map of labels to indicies dictionary
    model.class_to_idx = checkpoint['class_to_idx']
    
    # rebuilding classifier to attach to pretrained model
    classifier = nn.Sequential(nn.Dropout(0.3),
                                 nn.Linear(25088, 1000),
                                 nn.ReLU(),
                                 nn.Linear(1000, 102),
                                 nn.LogSoftmax(dim=1))
    
    # attaching the new classifier to the model
    model.classifier = classifier
    # getting the weights/params of the trained model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def test_pass(model, testloader, criterion, num_epochs, device):
    '''This should do a test pass using the data from the test data loader.
    If all goes according to plan I should still see a high accuracy on the
    testing data that the model has never seen.'''
    
    epochs = num_epochs
    
    # setting the model to evaluation mode
    model.eval()
    
    # sending model to cuda for testing quickly if it is available
    model.to(device)

    for e in range(epochs):  
        
        # instantiating test loss and test accuracy counters to zero
        test_loss = 0
        test_accuracy = 0

        # turning of saving gradients
        with torch.no_grad():
            # grabbing inputs and labels of test run
            for inputs, labels in tqdm_notebook(testloader):
                # sending to cuda
                inputs, labels = inputs.to(device), labels.to(device)
                # get output of model
                output = model(inputs)

                test_loss = criterion(output, labels)
                # update the test loss with new loss on each run
                test_loss += test_loss.item()

                # getting accuracy AKA doing some magic I don't quite understand yet
                # which gets the probablity outputs, the top classes and accuracy
                probs = torch.exp(output)
                top_prob, top_class = probs.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {e+1}/{epochs}... "
              f"Test Loss: {test_loss/len(testloader):.3f}... "
              f"Test accuracy: {test_accuracy/len(testloader):.3f}... ")

def process_image(image_path):
    '''
    This function should take an image path and process the image in a similar way
    that the images are processed from pytorch's torchvision. We can use numpy and pillow.
    '''

    # opens input image with pillow
    img = Image.open(image_path)
    
    # using thumbnail to figure resize based on whether the x or y axis is larger
    if img.size[0] > img.size[1]:
        img.thumbnail((500, 256))
    else:
        img.thumbnail((256, 500))
    
    # creating the center crop boundaries 
    left = (img.width-224)/2
    bottom = (img.height-224)/2
    right = left + 224
    top = bottom + 224
    
    # making that center crop!
    img = img.crop((left, bottom, right, top))
    
    # convert to numpy array - dividing by 256 to get array values between 0-1.
    img = np.array(img) / 256
    
    # the network expects the images to be normalized in a specific way
    means = np.array([0.485, 0.456, 0.406])
    standard_dev = np.array([0.229, 0.224, 0.225])
    img = (img - means)/standard_dev
    
    # transposing the image dimensions into the form pytorch models expect
    img = img.transpose((2,0,1))
    
    # returns processed image ready for putting through the model
    return(img)
              
def predict(img, cat_to_name, model, topk=1):
    '''This function takes an image path, a predictive model and
    returns the top probabilities, the top labels and the top flower
    names associated with those labels'''

    # taking numpy array to torch tensor 
    # need to read more on https://www.aiworkbox.com/lessons/convert-a-numpy-array-to-a-pytorch-tensor
    
    image = torch.from_numpy(img).type(torch.FloatTensor) 
    image = image.unsqueeze_(0)
    image = image.float()
    
    # no gradients again!
    with torch.no_grad():
        model.to("cpu")
        output = model(image)
        
        # grabbing the probabilities
        prob = torch.exp(output)
        
        # getting the probs and labels out in a form we can use
        top_prob, top_labels = prob.topk(int(topk), dim=1)
        
        # getting the top probs and labels in list formats
        top_prob = top_prob.detach().numpy().tolist()[0] 
        top_labels = top_labels.detach().numpy().tolist()[0]
        
        # bringing in the json mapping
        with open(cat_to_name, 'r') as f:
            cat_to_name = json.load(f)
        
        # some confusing dictionary hopping
        idx_to_class = {value: key for key, value in model.class_to_idx.items()}
        top_labels = [idx_to_class[labels] for labels in top_labels]
        top_classes = [cat_to_name[labels] for labels in top_labels]
        
        
    return(top_prob, top_labels, top_classes)
